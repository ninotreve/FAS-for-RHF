"""Opt-in physical RHF Newton on the mass-metric occupied quotient.

This solver excludes occupied rotations from the Krylov space. Forced coarse
systems retain the joint KKT solver: a general forcing breaks the physical
occupied-rotation symmetry and cannot be discarded by this projection.
"""
from time import perf_counter
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from .coarse import CoarseConfig, ReducedCoordinates, newton_linear_config
from .geometry import InvalidTrialState
from .records import model_work, work_difference, timed
from .scf import MolecularSolveResult


class HorizontalCoordinates:
    def __init__(self, model, state):
        model.validate(state)
        if not model.molecule.restricted:
            raise ValueError('Horizontal Newton requires physical closed-shell RHF')
        self.model, self.state = model, state
        self.phi = state.orbitals[0]
        self.shape, self.size = self.phi.shape, self.phi.size
        if np.linalg.norm(model.gram(self.phi)-np.eye(self.shape[1])) > 1e-9:
            raise ValueError('Horizontal Newton requires physical unit Gram')
        # Prefer the matrix-free tensor root even on small tensor grids.
        self.coordinates = ReducedCoordinates(model, 0)
        if self.coordinates.root is None:
            self.coordinates = ReducedCoordinates(model)
        if self.coordinates.root is None:
            raise ValueError('Horizontal Newton requires mass-whitened coordinates')
        self.U, _ = np.linalg.qr(self.white(self.phi).reshape(self.shape))

    def white(self, frame):
        return np.asarray(self.coordinates.root.T @ frame).ravel()

    def physical(self, vector):
        return self.coordinates.triangular_solve(vector.reshape(self.shape), transpose=True)

    def project(self, vector):
        x = vector.reshape(self.shape)
        return (x-self.U @ (self.U.T @ x)).ravel()

    def response(self, vector):
        h = self.project(vector)
        load = self.model.response_load(self.state, (self.physical(h),))[0]
        return self.project(self.coordinates.load_coordinates(load).ravel())+vector-h

    def precondition(self, vector):
        h = self.project(vector)
        load = self.coordinates.root @ h.reshape(self.shape)
        e = self.model.space.kinetic_inverse.relax(load)
        return self.project(self.white(e))+vector-h


@timed('Newton_horizontal_linear')
def linear_horizontal_step(model, state, config=None, *, residual=None):
    """Full induced response, kinetic preconditioner, independently checked RHS.

    The identity complement makes the ambient operator nonsingular without
    constructing a dense basis for the horizontal space. No curvature sign is
    assumed; failure or iteration exhaustion remains explicit in the report.
    """
    cfg = config or CoarseConfig()
    if (not 0 < cfg.krylov_tolerance < 1 or cfg.krylov_iterations < 1 or cfg.krylov_restart < 1):
        raise ValueError('Invalid horizontal Krylov configuration')
    c = HorizontalCoordinates(model, state)
    r = model.residual(state) if residual is None else residual
    if r.block_norms(model.space)['pair'] > 1e-9:
        raise ValueError('Horizontal Newton requires physical pair-consistent fields')
    rhs = -c.project(c.coordinates.load_coordinates(model.space.mass @ r.orbitals[0]).ravel())
    applications, iterations = 0, 0
    def action(v):
        nonlocal applications
        applications += 1
        return c.response(v)
    def callback(_):
        nonlocal iterations
        iterations += 1
    norm = np.linalg.norm(rhs)
    if not norm:
        step, info = np.zeros(c.size), 0
    else:
        A = LinearOperator((c.size, c.size), matvec=action, dtype=float)
        M = LinearOperator(A.shape, matvec=c.precondition, dtype=float)
        step, info = gmres(A, rhs, M=M, rtol=cfg.krylov_tolerance, atol=0.,
            restart=cfg.krylov_restart, maxiter=cfg.krylov_iterations,
            callback=callback, callback_type='legacy')
    direction = c.physical(c.project(step))
    # Check the actual projected direction, not an unprojected GMRES vector.
    defect = float(np.linalg.norm(action(c.white(direction))-rhs)/max(norm, np.finfo(float).tiny))
    horizontal = float(np.linalg.norm(c.phi.T @ (model.space.mass @ direction)))
    report = dict(engine='physical_RHF_horizontal_GMRES', iterations=iterations,
        response_applications=applications, exit_code=int(info), relative_residual=defect,
        target=cfg.krylov_tolerance, normal_direction_defect=horizontal,
        converged=bool(info == 0 and np.isfinite(defect) and defect <= cfg.krylov_tolerance))
    return direction, report


def solve_molecular_quotient(model, initial_state=None, config=None, callback=None):
    """Bounded retracted horizontal Newton, opt-in and physical RHF only.

    A decreasing full residual is required after restoring Gram, all pairs and
    multipliers. A failed line search stops explicitly; no hidden SCF/GS work.
    Residual convergence establishes a stationary endpoint, not a minimum.
    """
    cfg = config or CoarseConfig(gs_fallback=False)
    if (not np.isfinite(cfg.tolerance) or cfg.tolerance <= 0 or cfg.max_iterations < 1
        or not np.isfinite(cfg.constraint_trace_tolerance) or cfg.constraint_trace_tolerance <= 0):
        raise ValueError('Invalid horizontal Newton settings')
    if not model.molecule.restricted:
        raise ValueError('Horizontal Newton requires physical closed-shell RHF')
    started = perf_counter(); before = model_work(model)
    initial = model.initial_state() if initial_state is None else initial_state
    model.validate(initial)
    HorizontalCoordinates(model, initial)  # Reject forced/nonunit targets before rebuilding.
    state, residual = model.rebuild(initial.orbitals, return_residual=True)
    history = []
    for iteration in range(cfg.max_iterations+1):
        blocks = residual.block_norms(model.space)
        trace = float(2*abs(np.trace(residual.grams[0])))
        energy = model.energy(state)
        row = dict(iteration=iteration, residual=blocks['total'], residual_blocks=blocks,
            constraint_trace_error=trace, total_energy=energy.total,
            electronic_energy=energy.electronic, electron_count=model.electron_count(state))
        history.append(row)
        converged = blocks['total'] < cfg.tolerance and trace < cfg.constraint_trace_tolerance
        if callback is not None:
            callback(state, row)
        if converged or iteration == cfg.max_iterations:
            break
        direction, linear = linear_horizontal_step(model, state,
            newton_linear_config(cfg, blocks['total']), residual=residual)
        row['linear'], row['trials'] = linear, []
        accepted = False
        for weight in (1., .5, .25, .125, .0625, .03125):
            try:
                trial, defect = model.rebuild((state.orbitals[0]+weight*direction,), return_residual=True)
                merit = defect.norm(model.space)
            except InvalidTrialState as exc:
                row['trials'].append(dict(step=weight, rejected=str(exc)))
                continue
            row['trials'].append(dict(step=weight, residual=merit))
            if merit < blocks['total']:
                state, residual = trial, defect
                row['step'] = weight; accepted = True
                break
        if not accepted:
            row['exit'] = 'no_decreasing_step'
            break
    return MolecularSolveResult(state, converged, history, 'physical_RHF_horizontal_Newton',
        perf_counter()-started, work_difference(model, before))
