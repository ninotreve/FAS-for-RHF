"""Mesh-adjacency orbital block GS and forcing-preserving pair relaxation."""
import numpy as np
from scipy.sparse import eye, kron, csr_matrix
from .model import MolecularResidual
from .geometry import InvalidTrialState
from .records import timed
from .gs_scan import scan


@timed('GS_matrix')
def orbital_block_matrix(model, state, spin):
    space = model.space
    count = state.orbitals[spin].shape[1]
    if hasattr(space, 'orbital_block_matrix'):
        with model.work.measure('GS_tensor_assembly'):
            return space.orbital_block_matrix(model.hcore, model.hartree(state.pairs),
                                               state.pairs[spin], state.multipliers[spin])
    fixed = model.hcore+space.potential_matrix(model.hartree(state.pairs))
    matrix = kron(fixed, eye(count, format='csr'))-kron(space.mass.tocsr(), csr_matrix(state.multipliers[spin].T))
    for i in range(count):
        for j in range(count):
            index = csr_matrix(([1.], ([i], [j])), shape=(count, count))
            matrix -= kron(space.potential_matrix(state.pairs[spin][:, i, j]), index)
    return matrix.tobsr(blocksize=(count, count))


class FrozenOrbitalSweep:
    """Lazy matrices for weight trials of one unchanged GS starting state.

    A new instance is required after accepting a state. This is a local
    relaxation context, never a cache shared across sweeps, levels or calls.
    """
    def __init__(self, model, state):
        self.model, self.state = model, state
        self.matrices = {}

    def matrix(self, spin):
        if spin not in self.matrices:
            self.matrices[spin] = orbital_block_matrix(self.model, self.state, spin)
        else:
            self.model.work.add('GS_matrix_reuses')
        return self.matrices[spin]

    def __call__(self, forcing, weight, *, return_residual=False):
        return block_sweep(self.model, self.state, forcing, weight, _frozen=self,
                           return_residual=return_residual)


@timed('GS_sweep')
def block_sweep(model, state, forcing, weight, *, _frozen=None, return_residual=False):
    if _frozen is not None and (_frozen.model is not model or _frozen.state is not state):
        raise ValueError('Frozen GS matrices belong to a different model or starting state')
    space = model.space
    frames = []
    for spin, (p, b) in enumerate(zip(state.orbitals, forcing.orbitals)):
        count = p.shape[1]
        if count == 0:
            frames.append(p.copy()); continue
        matrix = orbital_block_matrix(model, state, spin) if _frozen is None else _frozen.matrix(spin)
        rhs = space.mass @ b
        x = p.copy()
        # Mesh vertices, rather than Cartesian parity, define the point blocks.
        with model.work.measure('GS_scan'):
            scan(matrix, rhs, x, weight)
        frames.append(model.retract(x, np.eye(count)+forcing.grams[spin]))
    # Rebuilding is the exact small-mesh pair solve; on AMG contexts the
    # family inverse has an explicit, independently checked tolerance.
    return model.rebuild(tuple(frames), forcing, retract=False, return_residual=return_residual)


@timed('relaxation')
def relax(model, state, forcing=None, sweeps=1, *, newton_fallback=True, diagnostics=None,
          newton_config=None, initial_residual=None, return_residual=False,tolerance=1e-13):
    """Try unit block steps first; damp only after a failed defect check."""
    forcing = MolecularResidual.zeros(state) if forcing is None else forcing
    if sweeps < 0 or tolerance <= 0:
        raise ValueError('Invalid block relaxation settings')
    residual = model.residual(state, forcing) if initial_residual is None else initial_residual
    for _ in range(sweeps):
        baseline = residual.norm(model.space)
        if baseline < tolerance:
            break
        accepted = state; accepted_residual = residual; improved = False
        attempts = []
        sweep = FrozenOrbitalSweep(model, state)
        for weight in (1., .5, .25, .125, .0625):
            try:
                trial, trial_residual = sweep(forcing, weight, return_residual=True)
            except InvalidTrialState as exc:
                attempts.append(dict(step=weight,rejected=str(exc)))
                continue
            merit = trial_residual.norm(model.space)
            attempts.append(dict(step=weight, residual=merit))
            if merit < baseline*(1-1e-5):
                accepted = trial; accepted_residual = trial_residual; improved = True
                break
        if diagnostics is not None:
            diagnostics.append(dict(method='orbital_block_GS', initial_residual=baseline,
                                    final_residual=accepted_residual.norm(model.space),
                                    step=weight if improved else 0., attempts=attempts))
        if not improved and newton_fallback:
            from .coarse import newton_relaxation_trial
            report = {}
            accepted, step, accepted_residual = newton_relaxation_trial(
                model, state, forcing, config=newton_config, report=report,
                initial_residual=residual, return_residual=True)
            if diagnostics is not None:
                diagnostics.append(dict(method='reduced_Newton_fallback', initial_residual=baseline,
                                        final_residual=accepted_residual.norm(model.space),
                                        step=step,**report))
        state, residual = accepted, accepted_residual
    return (state, residual) if return_residual else state
