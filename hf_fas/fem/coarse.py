"""Independent forced coarse solver with analytic reduced KKT derivatives.

Pair fields are eliminated with the active forcing. Newton corrections use
mass-whitened orbital coordinates and full-symmetric multiplier/Gram
coordinates. Gauge nullspaces are handled by a minimum-norm solve on small
levels; large levels use a matrix-free Krylov correction.
"""
from dataclasses import dataclass, replace
import numpy as np
from scipy.linalg import cholesky, solve_triangular, solve_sylvester
from scipy.sparse.linalg import LinearOperator, gmres
from .model import MolecularResidual, MolecularState
from .relaxation import relax
from .geometry import InvalidTrialState
from .records import timed


def pack_symmetric(matrix):
    i, j = np.triu_indices(len(matrix))
    return matrix[i, j]*np.where(i == j, 1., np.sqrt(2.))


def unpack_symmetric(values, count):
    result = np.zeros((count, count)); i, j = np.triu_indices(count)
    result[i, j] = values/np.where(i == j, 1., np.sqrt(2.))
    result[j, i] = result[i, j]
    return result


@dataclass(frozen=True)
class CoarseConfig:
    tolerance: float = 1e-10
    max_iterations: int = 30
    dense_limit: int = 0
    linear_solver: str = 'krylov'
    krylov_tolerance: float = 1e-8
    krylov_iterations: int = 200
    krylov_restart: int = 100
    forcing_ceiling: float | None = 1e-3
    gs_fallback: bool = True
    constraint_trace_tolerance: float = 1e-9


class ReducedCoordinates:
    def __init__(self, model, whiten_limit=512):
        self.model = model
        self.root = (cholesky(model.space.mass.toarray(), lower=True) if model.space.n_orbital <= whiten_limit
                     else getattr(model.space,'mass_cholesky',None))
        self.counts = model.molecule.occupations
        self.orbital_sizes = [model.space.n_orbital*n for n in self.counts]
        self.multiplier_sizes = [n*(n+1)//2 for n in self.counts]
        self.size = sum(self.orbital_sizes)+sum(self.multiplier_sizes)

    def directions(self, vector):
        offset = 0; orbitals = []; multipliers = []
        for count, size in zip(self.counts, self.orbital_sizes):
            white = vector[offset:offset+size].reshape(self.model.space.n_orbital, count)
            orbitals.append(self.triangular_solve(white,transpose=True) if self.root is not None else white.copy()); offset += size
        for count, size in zip(self.counts, self.multiplier_sizes):
            multipliers.append(unpack_symmetric(vector[offset:offset+size], count)); offset += size
        return tuple(orbitals), tuple(multipliers)

    def residual(self, state, forcing):
        space = self.model.space
        loads = self.model.action_load(state.orbitals, state.pairs)
        orbital = [self.load_coordinates(h-space.mass @ (p @ l+b)).ravel()
                   for p, l, h, b in zip(state.orbitals, state.multipliers, loads, forcing.orbitals)]
        gram = [pack_symmetric(self.model.gram(p)-np.eye(p.shape[1])-b) for p, b in zip(state.orbitals, forcing.grams)]
        return np.concatenate(orbital+gram)

    def jacobian_action(self, state, vector):
        e, dl = self.directions(vector)
        loads = self.model.response_load(state, e, dl)
        orbital = [self.load_coordinates(h).ravel() for h in loads]
        gram = [pack_symmetric(p.T @ (self.model.space.mass @ d)+d.T @ (self.model.space.mass @ p)) for p, d in zip(state.orbitals, e)]
        return np.concatenate(orbital+gram)

    def load_coordinates(self, load):
        return self.triangular_solve(load) if self.root is not None else self.model.space.mass_inverse.solve(load)

    def triangular_solve(self, values, *, transpose=False):
        if hasattr(self.root,'solve'):
            return self.root.solve(values,transpose=transpose)
        return solve_triangular(self.root.T if transpose else self.root,values,lower=not transpose)

    def metric_norm(self, vector):
        if self.root is not None:
            return float(np.linalg.norm(vector))
        offset = 0; squared = 0.
        for count,size in zip(self.counts,self.orbital_sizes):
            value = vector[offset:offset+size].reshape(self.model.space.n_orbital,count)
            offset += size
            squared += np.sum(value*(self.model.space.mass @ value))
        return float(np.sqrt(max(0.,squared+np.sum(vector[offset:]**2))))


def newton_linear_config(config, residual_norm):
    """Use one forcing rule for joint and retracted nonlinear Newton steps.

    Direct calls to linear_newton_step retain the strict configured target
    for qualification. None selects a fixed-tolerance nonlinear reference.
    """
    ceiling = config.forcing_ceiling
    if ceiling is None:
        return config
    if not config.krylov_tolerance <= ceiling < 1:
        raise ValueError('Newton forcing ceiling must be between the linear target and one')
    target = max(config.krylov_tolerance,min(ceiling,np.sqrt(residual_norm)))
    return replace(config,krylov_tolerance=float(target))


@timed('Newton_linear')
def linear_newton_step(model, state, coordinates, rhs, cfg):
    if cfg.linear_solver not in ('auto','dense','krylov'):
        raise ValueError('Unknown Newton linear solver')
    dense = cfg.linear_solver == 'dense' or (cfg.linear_solver == 'auto' and coordinates.size <= cfg.dense_limit)
    applications = 0
    def action(vector):
        nonlocal applications
        applications += 1
        return coordinates.jacobian_action(state,vector)
    if dense:
        jacobian = np.column_stack([action(x) for x in np.eye(coordinates.size)])
        step = np.linalg.lstsq(jacobian,rhs,rcond=1e-11)[0]
        defect = np.linalg.norm(jacobian @ step-rhs)
        info,iterations,engine = 0,0,'dense_minimum_norm'
    else:
        space = model.space; inverse = space.kinetic_inverse
        responses = tuple(inverse.relax(space.mass @ p) if p.shape[1] else p.copy() for p in state.orbitals)
        schur = tuple((p.T @ (space.mass @ y)+y.T @ (space.mass @ p))/2
                      for p,y in zip(state.orbitals,responses))
        @timed('Newton_preconditioner', lambda *args, **kwargs: model.work)
        def precondition(vector):
            offset = 0; frames = []
            for count,size in zip(coordinates.counts,coordinates.orbital_sizes):
                value = vector[offset:offset+size].reshape(space.n_orbital,count); offset += size
                load = coordinates.root @ value if coordinates.root is not None else space.mass @ value
                frames.append(inverse.relax(load) if count else value.copy())
            multipliers = []
            for p,e,y,h,count,size in zip(state.orbitals,frames,responses,schur,coordinates.counts,coordinates.multiplier_sizes):
                target = unpack_symmetric(vector[offset:offset+size],count); offset += size
                defect = target-p.T @ (space.mass @ e)-e.T @ (space.mass @ p)
                dl = solve_sylvester(h,h,defect) if count else target
                e += y @ dl
                multipliers.append(dl)
            packed = [((coordinates.root.T @ e) if coordinates.root is not None else e).ravel() for e in frames]
            return np.concatenate(packed+[pack_symmetric(dl) for dl in multipliers])
        operator = LinearOperator((coordinates.size,)*2,matvec=action,dtype=float)
        pre = LinearOperator((coordinates.size,)*2,matvec=precondition,dtype=float)
        iterations = 0
        def callback(_):
            nonlocal iterations
            iterations += 1
        tolerance=cfg.krylov_tolerance
        if coordinates.root is None:
            bounds = getattr(space, 'mass_spectral_bounds', None)
            if bounds is None:
                diagonal = space.mass.diagonal()
                bounds = (float(diagonal.min()/2), float(2.5*diagonal.max()))
            lower=min(bounds[0],1.)
            upper=max(bounds[1],1.)
            tolerance*=np.sqrt(lower/upper)
        step,info = gmres(operator,rhs,M=pre,rtol=tolerance,atol=0.,
            restart=cfg.krylov_restart,maxiter=cfg.krylov_iterations,callback=callback,callback_type='legacy')
        difference = action(step)-rhs; engine = 'preconditioned_GMRES'
        defect = coordinates.metric_norm(difference)
        rhs_norm = coordinates.metric_norm(rhs)
    relative = float(defect/max(np.linalg.norm(rhs) if dense else rhs_norm,np.finfo(float).tiny))
    return step,dict(engine=engine,exit_code=int(info),iterations=iterations,
        response_applications=applications,relative_residual=relative,
        target=cfg.krylov_tolerance,iteration_limit=cfg.krylov_iterations,
        converged=info == 0 and relative <= cfg.krylov_tolerance)


@timed('Newton_retracted_step')
def newton_relaxation_trial(model, state, forcing, *, config=None, report=None,
                            initial_residual=None, return_residual=False):
    """A safeguarded reduced Newton step, restoring the active Gram/pair target."""
    corrections = tuple(model.space.forcing_correction(b) for b in forcing.pairs)
    base = model.rebuild(state.orbitals, forcing, pair_corrections=corrections)
    cfg = config or CoarseConfig()
    coordinates = ReducedCoordinates(model,cfg.dense_limit)
    rhs = -coordinates.residual(base, forcing)
    linear_cfg = newton_linear_config(cfg,coordinates.metric_norm(rhs))
    step,linear = linear_newton_step(model,base,coordinates,rhs,linear_cfg)
    linear['reference_target'] = cfg.krylov_tolerance
    if report is not None:
        report['linear'] = linear
        report['trials'] = []
    directions, _ = coordinates.directions(step)
    residual = model.residual(state, forcing) if initial_residual is None else initial_residual
    baseline = residual.norm(model.space)
    for weight in (1., .5, .25, .125, .0625, .03125):
        frames = tuple(p+weight*d for p, d in zip(base.orbitals, directions))
        try:
            trial, trial_residual = model.rebuild(frames, forcing, pair_corrections=corrections,
                                                return_residual=True)
        except InvalidTrialState as exc:
            if report is not None: report['trials'].append(dict(step=weight,rejected=str(exc)))
            continue
        merit = trial_residual.norm(model.space)
        if report is not None: report['trials'].append(dict(step=weight,residual=merit))
        if merit < baseline:
            return (trial, weight, trial_residual) if return_residual else (trial, weight)
    return (state, 0., residual) if return_residual else (state, 0.)


@timed('Newton_KKT_solve')
def solve_forced_coarse(model, initial_state, forcing, config=None, *, callback=None,
                        rebuild_initial=True, return_residual=False):
    cfg = config or CoarseConfig()
    if cfg.tolerance <= 0 or cfg.constraint_trace_tolerance <= 0 or cfg.max_iterations < 1:
        raise ValueError('Invalid coarse solver settings')
    # This forcing is fixed for the entire coarse call, so its auxiliary
    # correction is solved once, not during every Newton/line-search trial.
    pair_corrections = tuple(model.space.forcing_correction(b) for b in forcing.pairs)
    if rebuild_initial:
        state, residual = model.rebuild(initial_state.orbitals, forcing,
            pair_corrections=pair_corrections, return_residual=True)
    else:
        state = initial_state.copy()
        residual = model.residual(state, forcing)
    model.validate(state)
    if not rebuild_initial and residual.block_norms(model.space)['pair'] > 1e-10:
        raise ValueError('Eliminated-pair Newton requires a pair-consistent initial state')
    coordinates = ReducedCoordinates(model, cfg.dense_limit)
    history = []
    def converged(residual):
        trace_error=sum(abs(w*np.trace(g)) for w,g in zip(model.molecule.weights,residual.grams))
        return residual.norm(model.space)<cfg.tolerance and trace_error<cfg.constraint_trace_tolerance
    for iteration in range(cfg.max_iterations+1):
        blocks = residual.block_norms(model.space); merit=blocks['total']
        trace_error=float(sum(abs(w*np.trace(g)) for w,g in zip(model.molecule.weights,residual.grams)))
        history.append(dict(iteration=iteration,residual=merit,residual_blocks=blocks,constraint_trace_error=trace_error))
        if callback is not None:
            callback(state, history[-1])
        if converged(residual) or iteration == cfg.max_iterations:
            break
        rhs = -coordinates.residual(state, forcing)
        linear_cfg = newton_linear_config(cfg,coordinates.metric_norm(rhs))
        step,linear = linear_newton_step(model,state,coordinates,rhs,linear_cfg)
        linear['reference_target'] = cfg.krylov_tolerance
        history[-1]['linear'] = linear
        history[-1]['trials'] = []
        e, dl = coordinates.directions(step)
        accepted = None
        for weight in (1., .5, .25, .125, .0625, .03125):
            frames = tuple(p+weight*d for p, d in zip(state.orbitals, e))
            try:
                trial, trial_residual = model.rebuild(frames, forcing, retract=False,
                    pair_corrections=pair_corrections, return_residual=True,
                    multipliers=tuple(l+weight*d for l,d in zip(state.multipliers,dl)))
            except InvalidTrialState as exc:
                history[-1]['trials'].append(dict(step=weight,rejected=str(exc)))
                continue
            value = trial_residual.norm(model.space)
            history[-1]['trials'].append(dict(step=weight,residual=value))
            if value < merit:
                accepted = trial; residual = trial_residual
                history[-1]['step'] = weight; break
        if accepted is None:
            if not cfg.gs_fallback:
                history[-1]['exit'] = 'no_decreasing_step'; break
            history[-1]['GS_fallback'] = []
            trial, trial_residual = relax(model, state.copy(), forcing, sweeps=2, newton_fallback=False,
                          diagnostics=history[-1]['GS_fallback'], initial_residual=residual,
                          return_residual=True,tolerance=cfg.tolerance)
            if trial_residual.norm(model.space) >= merit:
                history[-1]['exit'] = 'no_decreasing_step'; break
            accepted = trial; residual = trial_residual
        state = accepted
    final = residual.norm(model.space)
    success=bool(converged(residual))
    report = dict(converged=success, final_residual=final, history=history,
                  constraint_trace_tolerance=cfg.constraint_trace_tolerance,
                  stop_status='converged' if success else 'cap_or_stagnation')
    return (state, report, residual) if return_residual else (state, report)
