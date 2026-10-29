"""One low-rank commutator-DIIS SCF algorithm at every mesh size."""
from dataclasses import dataclass, field
from time import perf_counter
import warnings
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, lobpcg
from .linalg import SPDInverse, LinearSolveConfig
from .records import model_work, work_difference, timed


@dataclass(frozen=True)
class SCFConfig:
    tolerance: float = 1e-8
    max_iterations: int = 200
    damping: float = 1.
    diis_history: int = 6
    eigensolver: str = 'lobpcg'
    inner_tolerance: float = 1e-10
    inner_ceiling: float = 1e-4
    inner_factor: float = .05
    inner_iterations: int = 200
    constraint_trace_tolerance: float = 1e-9
    exchange_compression_tolerance: float | None = None


@dataclass
class MolecularSolveResult:
    state: object
    converged: bool
    history: list
    method: str
    elapsed_seconds: float
    work: dict = field(default_factory=dict)

    @property
    def final_residual(self):
        return self.history[-1]['residual']


def align(model, candidate, reference):
    if candidate.shape[1] == 0:
        return candidate.copy()
    u, _, vt = np.linalg.svd(candidate.T @ (model.space.mass @ reference))
    return candidate @ (u @ vt)


def commutator_inner(space, left, right):
    """Frobenius inner product of mass-whitened rank-2N commutators."""
    value = 0.
    for (a,b),(c,d) in zip(left,right):
        value += 2*(np.sum((a.T @ (space.mass @ c))*(b.T @ (space.mass @ d)))
                    -np.sum((a.T @ (space.mass @ d))*(b.T @ (space.mass @ c))))
    return float(value)


class Pulay:
    def __init__(self, count):
        self.count = count
        self.states, self.errors = [], []

    @timed('SCF_DIIS', lambda self, model, *args, **kwargs: model.work)
    def update(self, model, state):
        loads = model.action_load(state.orbitals,state.pairs)
        errors = tuple((model.space.mass_inverse.solve(h)-p @ lam,p.copy())
                       for h,p,lam in zip(loads,state.orbitals,state.multipliers))
        self.states.append(state.copy()); self.errors.append(errors)
        self.states, self.errors = self.states[-self.count:],self.errors[-self.count:]
        count = len(self.states)
        weights = np.zeros(count); weights[-1] = 1.
        if count > 1:
            gram = np.array([[commutator_inner(model.space,x,y) for y in self.errors] for x in self.errors])
            scale = np.trace(gram)/count
            if scale > 0:
                gram /= scale
                gram.flat[::count+1] += 1e-12
                system = np.ones((count+1,count+1)); system[:count,:count] = gram; system[-1,-1] = 0.
                rhs = np.zeros(count+1); rhs[-1] = 1.
                try:
                    weights = np.linalg.solve(system,rhs)[:-1]
                    weights /= weights.sum()
                except np.linalg.LinAlgError:
                    pass
        return tuple(self.states),weights


@timed('SCF_Fock_setup')
def prepare_fock_action(model, states, weights, spin, *, compression_tolerance=None):
    """Freeze density data for one eigensolve, without caching trial vectors.

    The caller owns this short-lived action. A new DIIS mixture always creates
    a new action; no data are attached to states or reused across SCF steps.
    """
    space = model.space
    hartree = sum(w*model.hartree(s.pairs) for w,s in zip(weights,states))
    occupied = np.concatenate([s.orbitals[spin] for s in states],axis=1)
    factors = np.concatenate([np.full(s.orbitals[spin].shape[1],w) for w,s in zip(weights,states)])
    full_occupied, full_factors = occupied, factors
    compression = None
    if compression_tolerance is not None:
        if not np.isfinite(compression_tolerance) or compression_tolerance <= 0:
            raise ValueError('Exchange compression tolerance must be positive and finite')
        with model.work.measure('SCF_density_compression'):
            occupied, factors, compression = compress_exchange_density(
                space, occupied, factors, compression_tolerance)
    model.work.add('mixed_fock_setups')
    def kernel(vectors, occupied, factors):
        matrix = vectors[:,None] if vectors.ndim == 1 else vectors
        out = model.hcore @ matrix+space.multiply_load(hartree,matrix)
        if len(factors):
            sources = space.source_products(occupied,matrix)
            loads = model.coulomb.integral_action(sources)
            out -= space.source_adjoint(occupied,loads*factors[None,:,None])
        return out[:,0] if vectors.ndim == 1 else out
    @timed('SCF_Fock', lambda *args, **kwargs: model.work)
    def action(vectors):
        model.work.add('mixed_fock_applications')
        return kernel(vectors,occupied,factors)
    @timed('SCF_Fock_validation', lambda *args, **kwargs: model.work)
    def original(vectors):
        model.work.add('mixed_fock_validation_applications')
        return kernel(vectors,full_occupied,full_factors)
    action.full_action = original if compression is not None else None
    action.compression = compression
    return action


def compress_exchange_density(space, occupied, factors, tolerance):
    """Signed spectral factorization of a small history density in the M metric.

    Negative DIIS occupations are retained. A numerical Gram-null cutoff and
    the requested density cutoff make this an approximation; every resulting
    eigensolve is checked with the original mixed Fock action before acceptance.
    """
    if not occupied.shape[1]:
        return occupied,factors,dict(input_columns=0,retained_columns=0,discarded_trace_norm=0.)
    gram=occupied.T@(space.mass@occupied)
    values,vectors=np.linalg.eigh((gram+gram.T)/2)
    floor=100*np.finfo(float).eps*max(1.,float(values[-1]))
    if values[0]<-floor or not np.isfinite(values).all():
        raise ValueError('History Gram is not finite positive semidefinite')
    keep=values>floor
    roots=np.sqrt(values[keep]);basis=vectors[:,keep]
    small=(roots[:,None]*(basis.T@(factors[:,None]*basis)))*roots[None,:]
    occupations,rotation=np.linalg.eigh((small+small.T)/2)
    retained=np.abs(occupations)>tolerance
    coefficients=(basis/roots[None,:])@rotation[:,retained]
    compact=occupied@coefficients
    return compact,occupations[retained],dict(input_columns=len(factors),
        retained_columns=int(retained.sum()),gram_rank=int(keep.sum()),gram_null_cutoff=floor,
        density_cutoff=tolerance,discarded_trace_norm=float(np.abs(occupations[~retained]).sum()),
        negative_retained_occupations=int(np.sum(occupations[retained]<0)),
        scope='Approximate signed history density; original mixed-Fock residual check is mandatory')


def mixed_fock_load(model, states, weights, spin, vectors):
    """One-shot Fock action; iterative callers should use prepare_fock_action."""
    return prepare_fock_action(model, states, weights, spin)(vectors)


@timed('SCF_eigensolve')
def occupied_eigensolve(model, action, frame, cfg, target, preconditioner):
    n,count = frame.shape
    if not count:
        return frame.copy(),dict(engine='empty',iterations=0,residual=0.,target=target,converged=True)
    operator = LinearOperator((n,n),matvec=action,matmat=action,dtype=float)
    messages = []
    if cfg.eigensolver == 'dense' or n < 5*count:
        matrix = action(np.eye(n))
        eigenvalues,vectors = eigh((matrix+matrix.T)/2,model.space.mass.toarray(),subset_by_index=(0,count-1))
        iterations,engine = 0,'dense_reference' if cfg.eigensolver == 'dense' else 'dense_small_block'
    else:
        inverse = LinearOperator((n,n),matvec=preconditioner.relax,matmat=preconditioner.relax,dtype=float)
        # P1 consistent mass satisfies M >= diag(M)/2. This bound translates
        # an aggregate inverse-mass residual target into Euclidean loads.
        bounds = getattr(model.space, 'mass_spectral_bounds', None)
        lower_mass = bounds[0] if bounds is not None else float(model.space.mass.diagonal().min()/2)
        tolerance = target*np.sqrt(lower_mass/count)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always',UserWarning)
            eigenvalues,vectors,history = lobpcg(operator,frame.copy(),B=model.space.mass,M=inverse,
                largest=False,tol=tolerance,maxiter=cfg.inner_iterations,retResidualNormsHistory=True)
        messages = [str(w.message) for w in captured][:3]
        iterations,engine = max(0,len(history)-3),'lobpcg'
    defect = action(vectors)-(model.space.mass @ vectors)*eigenvalues[None]
    residual = float(np.sqrt(max(0.,np.sum(defect*model.space.mass_inverse.solve(defect)))))
    full_action=getattr(action,'full_action',None)
    compression=getattr(action,'compression',None)
    if full_action is not None:
        full_defect=full_action(vectors)-(model.space.mass@vectors)*eigenvalues[None]
        full_residual=float(np.sqrt(max(0.,np.sum(full_defect*model.space.mass_inverse.solve(full_defect)))))
        checked=dict(compression,compressed_residual=residual,original_Fock_residual=full_residual,
                     fallback=not np.isfinite(full_residual) or full_residual>target)
        if checked['fallback']:
            result,report=occupied_eigensolve(model,full_action,frame,cfg,target,preconditioner)
            report['exchange_compression']=checked
            return result,report
        residual=full_residual
    else:
        checked=None
    return vectors,dict(engine=engine,iterations=iterations,residual=residual,target=target,
                        converged=residual <= target,warnings=messages,
                        **({'exchange_compression':checked} if checked is not None else {}))


def solve_molecular_scf(model, initial_state=None, config=None, callback=None):
    started = perf_counter(); before = model_work(model)
    cfg = config or SCFConfig()
    if (cfg.tolerance <= 0 or cfg.max_iterations < 1 or not 0 < cfg.damping <= 1
        or cfg.diis_history < 2 or cfg.eigensolver not in ('dense','lobpcg')
        or not 0 < cfg.inner_tolerance <= cfg.inner_ceiling or cfg.inner_factor <= 0
        or not np.isfinite(cfg.constraint_trace_tolerance) or cfg.constraint_trace_tolerance <= 0
        or (cfg.exchange_compression_tolerance is not None and
            (not np.isfinite(cfg.exchange_compression_tolerance) or cfg.exchange_compression_tolerance<=0))):
        raise ValueError('Invalid SCF settings')
    state = model.initial_state() if initial_state is None else initial_state.copy()
    model.validate(state)
    if any(not np.allclose(model.gram(p),np.eye(p.shape[1]),atol=1e-10,rtol=0) for p in state.orbitals):
        raise ValueError('SCF requires a unit-Gram physical initial state; use model.rebuild')
    space = model.space; mixer = Pulay(cfg.diis_history)
    preconditioner = space.kinetic_inverse
    history = []
    residual = model.residual(state)
    for iteration in range(cfg.max_iterations+1):
        blocks = residual.block_norms(space); energy = model.energy(state)
        trace_error = float(sum(abs(w*np.trace(g)) for w,g in
                                zip(model.molecule.weights, residual.grams)))
        converged = blocks['total'] < cfg.tolerance and trace_error < cfg.constraint_trace_tolerance
        record = dict(iteration=iteration,residual=blocks['total'],residual_blocks=blocks,
            constraint_trace_error=trace_error,
            electronic_energy=energy.electronic,total_energy=energy.total,
            electron_count=model.electron_count(state),coulomb_backend=model.coulomb.fingerprint,
            residual_convention='separate_consistent_mass_full_spin_pair_gram_v1',
            elapsed_seconds=perf_counter()-started)
        history.append(record)
        if converged or iteration == cfg.max_iterations:
            if callback is not None: callback(state,record)
            break
        states,weights = mixer.update(model,state)
        target = min(cfg.inner_ceiling,max(cfg.inner_tolerance,cfg.inner_factor*blocks['total']))
        candidate,records = [],[]
        for spin,p in enumerate(state.orbitals):
            action = prepare_fock_action(model, states, weights, spin,
                compression_tolerance=cfg.exchange_compression_tolerance)
            vectors,inner = occupied_eigensolve(model,action,p,cfg,target,preconditioner)
            candidate.append(align(model,vectors,p)); records.append(inner)
        record['inner'] = records
        record['diis_size'] = len(states)
        if callback is not None: callback(state,record)
        frames = tuple((1-cfg.damping)*p+cfg.damping*c for p,c in zip(state.orbitals,candidate))
        state, residual = model.rebuild(frames, return_residual=True)
    return MolecularSolveResult(state,converged,history,'commutator_DIIS_SCF',
                                perf_counter()-started,work_difference(model,before))
