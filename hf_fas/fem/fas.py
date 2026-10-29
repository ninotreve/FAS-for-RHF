"""Complete forced molecular FAS with independent pair-space transfers."""
from dataclasses import dataclass, field
from time import perf_counter
import numpy as np
from .model import MolecularResidual, symmetric_power
from .transfer import FEMTransfer
from .relaxation import relax
from .coarse import CoarseConfig, solve_forced_coarse
from .scf import MolecularSolveResult
from .geometry import InvalidTrialState, checked_alignment
from .records import model_work, work_difference, timed


@dataclass(frozen=True)
class FASConfig:
    tolerance: float = 1e-8
    max_cycles: int = 100
    pre_sweeps: int = 2
    post_sweeps: int = 2
    correction: str = 'geometric'
    newton_smoother_fallback: bool = True
    coarse: CoarseConfig = field(default_factory=CoarseConfig)


class FEMHierarchy:
    def __init__(self, models):
        if not models:
            raise ValueError('At least one FEM level is required')
        self.models = tuple(models)
        self.coarse_records = []
        self.smoother_records = []
        self.cycle_records = []
        self.current_cycle = 0
        self.transfers = tuple(FEMTransfer(coarse, fine) for coarse, fine in zip(models[:-1], models[1:]))
        config = models[0].coulomb.config
        for model in models[1:]:
            if model.coulomb.config != config:
                raise ValueError('Every hierarchy level must use the same Coulomb backend policy')
            if model.space.config != models[0].space.config:
                raise ValueError('Every level must share the mass, pair-scaling and nuclear quadrature policy')
            if not np.array_equal(model.molecule.atomic_numbers, models[0].molecule.atomic_numbers) or not np.array_equal(model.molecule.centers, models[0].molecule.centers):
                raise ValueError('Molecular geometry must match on all levels')

    def _physical_converged(self, level, forcing, residual, cfg):
        """Early exit applies only to the finest zero-forcing physical system.

        Forced coarse equations must not inherit the physical stopping test.
        The weighted Gram trace protects electron count, including many orbitals.
        """
        if level != len(self.models)-1 or residual.norm(self.models[level].space) >= cfg.tolerance:
            return False
        if any(np.any(x) for group in (forcing.orbitals, forcing.pairs, forcing.grams) for x in group):
            return False
        trace_error = sum(abs(w*np.trace(g)) for w,g in
                          zip(self.models[level].molecule.weights, residual.grams))
        return trace_error < cfg.coarse.constraint_trace_tolerance

    def _relax(self, level, state, forcing, sweeps, cfg, phase, residual=None):
        offset = len(self.smoother_records)
        with self.models[level].work.measure('pre_smoothing' if phase == 'pre' else 'post_smoothing'):
            output = relax(self.models[level],state,forcing,sweeps,
                newton_fallback=cfg.newton_smoother_fallback,diagnostics=self.smoother_records,
                newton_config=cfg.coarse, initial_residual=residual, return_residual=True,
                tolerance=cfg.tolerance)
        for record in self.smoother_records[offset:]:
            record.update(level=level,cycle=self.current_cycle,phase=phase)
        return output

    @timed('FAS_cycle', lambda self, level, *args, **kwargs: self.models[level].work)
    def cycle(self, level, state, forcing, cfg, *, initial_residual=None, return_residual=False):
        model = self.models[level]
        def finish(state, residual):
            return (state, residual) if return_residual else state
        if level == 0:
            solved, report, residual = solve_forced_coarse(model, state, forcing, cfg.coarse, return_residual=True)
            report.update(level=level,cycle=self.current_cycle)
            self.coarse_records.append(report)
            return finish(solved, residual)
        state, residual = self._relax(level,state,forcing,cfg.pre_sweeps,cfg,'pre',initial_residual)
        base = state.copy(); merit = residual.norm(model.space)
        record = dict(level=level,cycle=self.current_cycle,pre_residual=merit,trials=[])
        self.cycle_records.append(record)
        if self._physical_converged(level, forcing, residual, cfg):
            record.update(selected_step=0., final_residual=merit, fallback=False, stopped_after='pre')
            return finish(state, residual)
        transfer = self.transfers[level-1]
        restricted, coarse_rhs = transfer.coarse_equation(state, forcing, fine_residual=residual)
        coarse_solution = self.cycle(level-1, restricted.copy(), coarse_rhs, cfg)
        coarse = self.models[level-1]
        try:
            directions = []
            for p, z, fine_p, bg in zip(restricted.orbitals, coarse_solution.orbitals, base.orbitals, forcing.grams):
                if p.shape[1] == 0:
                    directions.append(fine_p.copy()); continue
                yhat = p @ symmetric_power(coarse.gram(p), -.5)
                zhat = z @ symmetric_power(coarse.gram(z), -.5)
                rotation = checked_alignment(yhat,zhat,coarse.space.mass)
                aligned = z @ rotation
                if cfg.correction == 'component':
                    direction = transfer.orbital_prolongation @ (aligned-p)
                else:
                    chord = zhat @ rotation-yhat
                    horizontal = chord-yhat @ (yhat.T @ (coarse.space.mass @ chord))
                    prolonged = transfer.orbital_prolongation @ horizontal
                    target = np.eye(p.shape[1])+bg
                    fhat = fine_p @ symmetric_power(target, -.5)
                    direction = (prolonged-fhat @ (fhat.T @ (model.space.mass @ prolonged))) @ symmetric_power(target, .5)
                directions.append(direction)
        except InvalidTrialState as exc:
            post, post_residual = self._relax(level,base.copy(),forcing,cfg.post_sweeps,cfg,'post_only',residual)
            record.update(correction_rejected=str(exc),selected_step=0.,fallback=True,
                          final_residual=post_residual.norm(model.space))
            return finish(post, post_residual)
        trials = []
        for weight in (1., .5, .25, .125, .0625):
            frames = tuple(p+weight*d for p, d in zip(base.orbitals, directions))
            try:
                trial, trial_residual = model.rebuild(frames, forcing, return_residual=True)
            except InvalidTrialState as exc:
                record['trials'].append(dict(step=weight,rejected=str(exc)))
                continue
            trial, trial_residual = self._relax(level,trial,forcing,cfg.post_sweeps,cfg,f'post_{weight}',trial_residual)
            value = trial_residual.norm(model.space)
            trials.append((value, trial, trial_residual))
            record['trials'].append(dict(step=weight,residual=value))
            physical_stop = self._physical_converged(level, forcing, trial_residual, cfg)
            if value <= merit or physical_stop:
                record.update(selected_step=weight,final_residual=value,fallback=False)
                if physical_stop:
                    record['stopped_after'] = 'post'
                return finish(trial, trial_residual)
        post, post_residual = self._relax(level,base.copy(),forcing,cfg.post_sweeps,cfg,'post_only',residual)
        trials.append((post_residual.norm(model.space), post, post_residual))
        record['trials'].append(dict(step=0.,residual=trials[-1][0]))
        selected = min(range(len(trials)),key=lambda k:trials[k][0])
        valid = [r for r in record['trials'] if 'residual' in r]
        record.update(selected_step=valid[selected]['step'],final_residual=trials[selected][0],fallback=True)
        return finish(trials[selected][1], trials[selected][2])


def solve_molecular_fas(hierarchy, initial_state=None, config=None, callback=None):
    started = perf_counter()
    cfg = config or FASConfig()
    if cfg.correction not in ('geometric', 'component') or cfg.tolerance <= 0 or cfg.max_cycles < 1:
        raise ValueError('Invalid molecular FAS configuration')
    model = hierarchy.models[-1]
    before = [model_work(m) for m in hierarchy.models]
    state = model.initial_state() if initial_state is None else initial_state.copy()
    model.validate(state)
    forcing = MolecularResidual.zeros(state)
    residual = model.residual(state)
    history = []
    for cycle in range(cfg.max_cycles+1):
        blocks = residual.block_norms(model.space); merit = blocks['total']
        energy = model.energy(state)
        record = dict(cycle=cycle, residual=merit, residual_blocks=blocks,electronic_energy=energy.electronic,
                      total_energy=energy.total, electron_count=model.electron_count(state),
                      coulomb_backend=model.coulomb.fingerprint,
                      residual_convention='separate_consistent_mass_full_spin_pair_gram_v1')
        history.append(record)
        if callback is not None:
            callback(state, record)
        converged = hierarchy._physical_converged(len(hierarchy.models)-1, forcing, residual, cfg)
        if converged or cycle == cfg.max_cycles:
            break
        hierarchy.current_cycle = cycle+1
        state, residual = hierarchy.cycle(len(hierarchy.models)-1, state, forcing, cfg,
                                         initial_residual=residual, return_residual=True)
    return MolecularSolveResult(state, converged, history, 'complete_molecular_FAS',
        perf_counter()-started,dict(levels=[work_difference(m,b) for m,b in zip(hierarchy.models,before)],
                                   level_visits=len(hierarchy.cycle_records)+len(hierarchy.coarse_records)))
