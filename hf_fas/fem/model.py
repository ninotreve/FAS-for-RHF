"""Complete molecular RHF/UHF equations in distinct orbital and pair spaces."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh, solve_sylvester
from .geometry import checked_gram, InvalidTrialState, symmetric_trial_matrix
from .records import WorkCounter, timed


def symmetric_power(matrix, exponent):
    if matrix.size == 0:
        return matrix.copy()
    if not np.isfinite(matrix).all() or np.linalg.norm(matrix-matrix.T,ord='fro') > 1e-11*len(matrix):
        raise ValueError('Gram target is not symmetric')
    values, vectors = eigh(matrix)
    if values[0] <= 0 or values[-1]/values[0] > 1e12:
        raise ValueError('Non-SPD or ill-conditioned Gram matrix; never clip')
    result=(vectors*values**exponent) @ vectors.T
    return .5*result+.5*result.T


@dataclass
class MolecularState:
    orbitals: tuple[np.ndarray, ...]
    pairs: tuple[np.ndarray, ...]
    multipliers: tuple[np.ndarray, ...]

    def copy(self):
        return MolecularState(tuple(x.copy() for x in self.orbitals),
                              tuple(x.copy() for x in self.pairs),
                              tuple(x.copy() for x in self.multipliers))

    def rotate(self, rotations):
        return MolecularState(tuple(p @ q for p, q in zip(self.orbitals, rotations)),
                              tuple(np.einsum('ir,aij,js->ars', q, w, q) for w, q in zip(self.pairs, rotations)),
                              tuple(q.T @ l @ q for l, q in zip(self.multipliers, rotations)))


@dataclass
class MolecularResidual:
    orbitals: tuple[np.ndarray, ...]
    pairs: tuple[np.ndarray, ...]
    grams: tuple[np.ndarray, ...]

    @classmethod
    def zeros(cls, state):
        return cls(tuple(np.zeros_like(p) for p in state.orbitals),
                   tuple(np.zeros_like(w) for w in state.pairs),
                   tuple(np.zeros_like(l) for l in state.multipliers))

    def copy(self):
        return MolecularResidual(*(tuple(x.copy() for x in group) for group in (self.orbitals, self.pairs, self.grams)))

    def __add__(self, other):
        return MolecularResidual(*(tuple(a+b for a, b in zip(left, right)) for left, right in zip((self.orbitals, self.pairs, self.grams), (other.orbitals, other.pairs, other.grams))))

    def __sub__(self, other):
        return MolecularResidual(*(tuple(a-b for a, b in zip(left, right)) for left, right in zip((self.orbitals, self.pairs, self.grams), (other.orbitals, other.pairs, other.grams))))

    def block_norms(self, space):
        orbital = sum(float(np.sum(p*(space.mass @ p))) for p in self.orbitals)
        pair = sum(float(np.sum(w.reshape(space.n_pair, -1)*(space.pair_mass @ w.reshape(space.n_pair, -1)))) for w in self.pairs)
        gram = sum(float(np.sum(g*g)) for g in self.grams)
        if not np.isfinite((orbital, pair, gram)).all():
            raise InvalidTrialState('Nonfinite molecular residual norm')
        return dict(orbital=float(np.sqrt(max(0.,orbital))),
                    pair=float(np.sqrt(max(0.,pair))), gram=float(np.sqrt(max(0.,gram))),
                    total=float(np.sqrt(max(0.,orbital+pair+gram))))

    def norm(self, space):
        return self.block_norms(space)['total']

    def rotate(self, rotations):
        return MolecularResidual(tuple(p @ q for p, q in zip(self.orbitals, rotations)),
                                 tuple(np.einsum('ir,aij,js->ars', q, w, q) for w, q in zip(self.pairs, rotations)),
                                 tuple(q.T @ g @ q for g, q in zip(self.grams, rotations)))


@dataclass(frozen=True)
class MolecularEnergy:
    one_electron: float
    direct: float
    exchange: float
    nuclear: float

    @property
    def electronic(self):
        return self.one_electron+self.direct+self.exchange

    @property
    def total(self):
        return self.electronic+self.nuclear


class MolecularFEMModel:
    def __init__(self, space, molecule, coulomb=None):
        self.space, self.molecule = space, molecule
        self.work = WorkCounter()
        if max(molecule.occupations) > space.n_orbital:
            raise ValueError('More occupied orbitals than interior FEM degrees of freedom; refine mesh')
        self.coulomb = coulomb if coulomb is not None else space.make_coulomb()
        if self.coulomb.space is not space:
            raise ValueError('Coulomb and orbital model must share the same FEM context')
        nuclear = space.nuclear_matrix(molecule.centers, molecule.atomic_numbers)
        stiffness = space.stiffness.tocsr() if hasattr(space.stiffness, 'tocsr') else space.stiffness
        self.hcore = .5*stiffness+nuclear

    def validate(self, state):
        if not (len(state.orbitals) == len(state.pairs) == len(state.multipliers) == len(self.molecule.occupations)):
            raise ValueError('Incorrect number of spin channels')
        for p, w, l, count in zip(state.orbitals, state.pairs, state.multipliers, self.molecule.occupations):
            if p.shape != (self.space.n_orbital, count) or w.shape != (self.space.n_pair, count, count) or l.shape != (count, count):
                raise ValueError('Molecular state dimensions differ from FEM spaces/occupations')
            if not all(np.isfinite(x).all() for x in (p, w, l)):
                raise ValueError('Nonfinite molecular state')
            # Frobenius skew norms preserve the symmetry tolerance under
            # occupied rotations; check each spatial pair matrix separately.
            if (np.any(np.linalg.norm(w-w.swapaxes(-1, -2),axis=(-2,-1)) > 1e-11*count)
                or np.linalg.norm(l-l.T,ord='fro') > 1e-11*count):
                raise ValueError('Pair family and multipliers must be symmetric')

    def gram(self, orbitals):
        return orbitals.T @ (self.space.mass @ orbitals)

    @timed('Gram_restore')
    def retract(self, orbitals, target=None):
        if orbitals.shape[1] == 0:
            return orbitals.copy()
        gram = checked_gram(self.space.mass, orbitals)
        target = np.eye(len(gram)) if target is None else target
        symmetric_power(target,.5)  # Invalid caller targets are configuration errors.
        root = symmetric_power(gram, .5); inverse = symmetric_power(gram, -.5)
        try:
            middle=symmetric_trial_matrix(root @ target @ root)
            right = symmetric_trial_matrix(inverse @ symmetric_power(middle, .5) @ inverse)
        except ValueError as exc:
            raise InvalidTrialState('Numerically inadmissible Gram restoration') from exc
        restored=orbitals @ right
        error=np.linalg.norm(self.gram(restored)-target,ord='fro')
        if not np.isfinite(error) or error>1e-10*max(1.,np.linalg.norm(target,ord='fro')):
            raise InvalidTrialState('Trial Gram restoration failed its numerical target check')
        return restored

    def hartree(self, pairs):
        return sum(weight*np.trace(w, axis1=1, axis2=2) for w, weight in zip(pairs, self.molecule.weights))

    @timed('orbital_action')
    def action_load(self, orbitals, pairs):
        self.work.add('orbital_actions')
        j = self.hartree(pairs)
        action = []
        for p, w in zip(orbitals, pairs):
            out = self.hcore @ p+self.space.multiply_load(j, p)-self.space.exchange_load(w, p)
            action.append(out)
        return tuple(action)

    def update_multipliers(self, state, forcing=None):
        self._update_multipliers(state, forcing, self.action_load(state.orbitals, state.pairs))

    def _update_multipliers(self, state, forcing, actions):
        forcing = MolecularResidual.zeros(state) if forcing is None else forcing
        values = []
        for p, action, b in zip(state.orbitals, actions, forcing.orbitals):
            if p.shape[1] == 0:
                values.append(np.empty((0, 0))); continue
            gram = self.gram(p); k = p.T @ (action-self.space.mass @ b)
            lam = solve_sylvester(gram, gram, k+k.T)
            values.append((lam+lam.T)/2)
        state.multipliers = tuple(values)

    @timed('pair_rebuild')
    def rebuild(self, orbitals, forcing=None, *, retract=True, pair_corrections=None,
                return_residual=False, multipliers=None):
        """Build one candidate, optionally evaluating its complete residual.

        The unforced pair fields and orbital actions are shared only within
        this call. No cached data is attached to the mutable returned state.
        Explicit multipliers retain joint Newton's prescribed KKT update.
        """
        self.work.add('reconstructions')
        counts = self.molecule.occupations
        if len(orbitals) != len(counts):
            raise ValueError('Incorrect orbital spin family')
        targets = [np.eye(n) for n in counts] if forcing is None else [np.eye(n)+g for n, g in zip(counts, forcing.grams)]
        frames = tuple(self.retract(p, target) if retract else np.asarray(p).copy() for p, target in zip(orbitals, targets))
        if not retract:
            for frame in frames:
                checked_gram(self.space.mass, frame)
        base_fields = tuple(self.coulomb(self.space.sources(p)) for p in frames)
        fields = base_fields
        if forcing is not None:
            corrections = tuple(self.space.forcing_correction(b) for b in forcing.pairs) if pair_corrections is None else pair_corrections
            fields = tuple(w+c for w, c in zip(fields, corrections))
        elif pair_corrections is not None:
            raise ValueError('Cached pair forcing corrections require an active forcing')
        state = MolecularState(frames, fields, tuple(np.zeros((n, n)) for n in counts))
        actions = self.action_load(state.orbitals, state.pairs)
        if multipliers is None:
            self._update_multipliers(state, forcing, actions)
        else:
            state.multipliers = tuple(np.asarray(l).copy() for l in multipliers)
        self.validate(state)
        if return_residual:
            result = self._operator(state, actions, base_fields)
            return state, result if forcing is None else result-forcing
        return state

    def initial_state(self, seed=20261014):
        # Generalized eigenvectors respect the consistent, non-diagonal mass.
        if self.space.n_orbital > 512 and max(self.molecule.occupations) < self.space.n_orbital-1:
            from scipy.sparse.linalg import eigsh, LinearOperator
            count = max(self.molecule.occupations)
            if count == 0:
                vectors = np.empty((self.space.n_orbital, 0))
            else:
                _, vectors = eigsh(self.hcore, k=count, M=self.space.mass, which='SA', tol=1e-10,
                                   Minv=LinearOperator(
                                       self.space.mass.shape,matvec=self.space.mass_inverse.solve,matmat=self.space.mass_inverse.solve,dtype=float),
                                   v0=np.random.default_rng(seed).normal(size=self.space.n_orbital))
        else:
            _, vectors = eigh(self.hcore.toarray(), self.space.mass.toarray())
        return self.rebuild(tuple(vectors[:, :n].copy() for n in self.molecule.occupations))

    def operator(self, state):
        return self._operator(state)

    @timed('residual')
    def _operator(self, state, actions=None, fields=None):
        self.work.add('residual_evaluations')
        self.validate(state)
        if actions is None:
            actions = self.action_load(state.orbitals, state.pairs)
        orbital = tuple(self.space.mass_inverse.solve(h-self.space.mass @ (p @ l)) for p, l, h in zip(state.orbitals, state.multipliers, actions))
        if fields is None:
            fields = tuple(self.coulomb(self.space.sources(p)) for p in state.orbitals)
        pair = tuple(self.space.pair_scale(w-field) for w, field in zip(state.pairs, fields))
        gram = tuple(self.gram(p)-np.eye(p.shape[1]) for p in state.orbitals)
        return MolecularResidual(orbital, pair, gram)

    def residual(self, state, forcing=None):
        result = self.operator(state)
        return result if forcing is None else result-forcing

    @timed('energy')
    def energy(self, state):
        self.work.add('energy_evaluations')
        # Evaluate the orbital functional through the same C, even for a
        # caller-supplied off-manifold pair field. Forced-state merit uses
        # residuals, never this physical energy.
        sources = tuple(self.space.sources(p) for p in state.orbitals)
        fields = tuple(self.coulomb(s) for s in sources)
        density = sum(weight*np.trace(s, axis1=1, axis2=2) for s, weight in zip(sources, self.molecule.weights))
        j = self.hartree(fields)
        one = sum(weight*float(np.sum(p*(self.hcore @ p))) for p, weight in zip(state.orbitals, self.molecule.weights))
        direct = .5*float(density @ (self.space.pair_mass @ j))
        exchange = -.5*sum(weight*float(np.sum(s.reshape(self.space.n_pair, -1)*(self.space.pair_mass @ w.reshape(self.space.n_pair, -1)))) for s, w, weight in zip(sources, fields, self.molecule.weights))
        return MolecularEnergy(one, direct, exchange, self.molecule.nuclear_repulsion)

    @timed('Fock_action')
    def fock_load(self, state, spin, vectors):
        self.work.add('fock_applications')
        p = state.orbitals[spin]
        matrix = vectors[:, None] if vectors.ndim == 1 else vectors
        out = self.hcore @ matrix+self.space.multiply_load(self.hartree(state.pairs), matrix)
        if matrix.shape[1] == 0:
            return out
        # Group trial exchange sources to amortize tree construction. Bound
        # temporary storage by one batch (or one wide trial block), independent
        # of the total number of occupied orbitals.
        block = max(1, self.coulomb.config.assembly_batch//matrix.shape[1])
        for start in range(0, p.shape[1], block):
            occupied = p[:, start:start+block]
            source = self.space.source_products(occupied,matrix)
            load = self.coulomb.integral_action(source)
            out -= self.space.source_adjoint(occupied,load)
        return out[:, 0] if vectors.ndim == 1 else out

    @timed('response')
    def response_load(self, state, directions, multiplier_directions=None):
        self.work.add('response_applications')
        derivative_fields = []
        for p, e in zip(state.orbitals, directions):
            derivative_fields.append(self.coulomb(self.space.source_products(p,e)+self.space.source_products(e,p)))
        dj = self.hartree(derivative_fields)
        result = []
        for spin, (p, e, l, h) in enumerate(zip(state.orbitals, directions, state.multipliers, self.action_load(directions, state.pairs))):
            value = h-self.space.mass @ (e @ l)+self.space.multiply_load(dj, p)
            if multiplier_directions is not None:
                value -= self.space.mass @ (p @ multiplier_directions[spin])
            value -= self.space.exchange_load(derivative_fields[spin], p)
            result.append(value)
        return tuple(result)

    def horizontal_response(self, state, directions):
        projected = tuple(e-p @ (p.T @ (self.space.mass @ e)) for p, e in zip(state.orbitals, directions))
        response = tuple(self.space.mass_inverse.solve(h) for h in self.response_load(state, projected))
        return tuple(h-p @ (p.T @ (self.space.mass @ h)) for p, h in zip(state.orbitals, response))

    def electron_count(self, state):
        return float(sum(weight*np.trace(self.gram(p)) for p, weight in zip(state.orbitals, self.molecule.weights)))
