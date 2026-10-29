"""Consistent tensor masses and exact DST/DCT shifted-elliptic inverses."""
import numpy as np
from scipy.fft import dstn, dctn
from scipy.linalg import cholesky_banded, cho_solve_banded, solve_banded
from scipy.sparse import kron, diags
from scipy.sparse.linalg import LinearOperator
from hf_fas.fem.records import WorkCounter, timed


def axis_action(matrix, values, axis):
    moved = np.moveaxis(values, axis, 0)
    output = matrix @ moved.reshape(moved.shape[0], -1)
    return np.moveaxis(output.reshape((matrix.shape[0],)+moved.shape[1:]), 0, axis)


class TensorProduct(LinearOperator):
    def __init__(self, factors):
        self.factors = tuple(f.tocsr() for f in factors)
        self.input_shape = tuple(f.shape[1] for f in self.factors)
        self.output_shape = tuple(f.shape[0] for f in self.factors)
        self._sparse = None
        super().__init__(dtype=np.dtype(float), shape=(int(np.prod(self.output_shape)), int(np.prod(self.input_shape))))

    def _matmat(self, values):
        shaped = np.asarray(values).reshape(self.input_shape+(-1,))
        for axis, matrix in enumerate(self.factors):
            shaped = axis_action(matrix, shaped, axis)
        return shaped.reshape(self.shape[0], -1)

    def _matvec(self, values):
        return self._matmat(values.reshape(self.shape[1],1))[:, 0]

    def _adjoint(self):
        return TensorProduct(tuple(f.T for f in self.factors))

    def diagonal(self):
        result = self.factors[0].diagonal()
        for f in self.factors[1:]:
            result = np.kron(result, f.diagonal())
        return result

    def tocsr(self, *, cache=True):
        if not cache:
            return kron(kron(self.factors[0], self.factors[1], format='csr'), self.factors[2], format='csr')
        if self._sparse is None:
            self._sparse = self.tocsr(cache=False)
        return self._sparse

    def toarray(self):
        return self.tocsr().toarray()


class TensorSum(LinearOperator):
    def __init__(self, terms):
        self.terms = tuple(terms)
        super().__init__(dtype=np.dtype(float), shape=self.terms[0][1].shape)

    def _matmat(self, values):
        return sum(scale*(operator @ values) for scale, operator in self.terms)

    def _matvec(self, values):
        return self._matmat(values.reshape(self.shape[1],1))[:, 0]

    def _adjoint(self):
        return TensorSum(tuple((a, b.T) for a, b in self.terms))

    def diagonal(self):
        return sum(a*b.diagonal() for a, b in self.terms)

    def tocsr(self):
        # A temporary Kronecker sum must not retain every expanded term.
        return sum(a*(b.tocsr(cache=False) if isinstance(b, TensorProduct) else b.tocsr())
                   for a, b in self.terms).tocsr()

    def toarray(self):
        return self.tocsr().toarray()


class TensorCholesky(TensorProduct):
    """Matrix-free L in M=L L^T, including exact triangular solves."""
    def __init__(self, band):
        self.band = band
        factor=diags((band[0],band[1,:-1]),(0,-1),format='csr')
        super().__init__((factor,)*3)

    def solve(self, rhs, *, transpose=False):
        shape=rhs.shape
        if not np.any(rhs):
            return np.zeros_like(rhs)
        n=self.factors[0].shape[0]
        values=np.asarray(rhs).reshape((n,)*3+(-1,))
        band=self.band
        if transpose:
            band=np.zeros_like(band);band[1]=self.band[0];band[0,1:]=self.band[1,:-1]
        for axis in range(3):
            moved=np.moveaxis(values,axis,0)
            solved=solve_banded((0,1) if transpose else (1,0),band,moved.reshape(n,-1))
            values=np.moveaxis(solved.reshape(moved.shape),0,axis)
        return values.reshape(shape)


class TensorMassInverse:
    def __init__(self, factor):
        self.factor = factor.tocsr()
        self.size = factor.shape[0]
        self.work = WorkCounter(); self.last_residual = 0.
        self.matrix = TensorProduct((factor,)*3)
        with self.work.measure('setup'):
            band = np.zeros((2, self.size)); band[0] = factor.diagonal()
            band[1, :-1] = factor.diagonal(-1)
            self.cholesky = cholesky_banded(band, lower=True)
            self.root = TensorCholesky(self.cholesky)

    @timed('solve')
    def solve(self, rhs):
        self.work.add('solves')
        rhs = np.asarray(rhs, dtype=float); shape = rhs.shape
        if rhs.ndim == 0 or shape[0] != self.size**3:
            raise ValueError('Tensor mass dimension mismatch')
        if not np.isfinite(rhs).all():
            raise ValueError('Nonfinite tensor mass RHS')
        norm = np.linalg.norm(rhs)
        if not norm:
            self.last_residual = 0.
            return np.zeros_like(rhs)
        values = rhs.reshape((self.size,)*3+(-1,)).copy()
        for axis in range(3):
            moved = np.moveaxis(values, axis, 0)
            solved = cho_solve_banded((self.cholesky, True), moved.reshape(self.size, -1))
            values = np.moveaxis(solved.reshape(moved.shape), 0, axis)
        result = values.reshape(shape)
        self.last_residual = float(np.linalg.norm(self.matrix @ result.reshape(shape[0],-1)-rhs.reshape(shape[0],-1))/norm)
        if not np.isfinite(self.last_residual) or self.last_residual > 1e-10:
            raise RuntimeError('Tensor mass inverse failed its true residual check')
        return result


class TensorEllipticInverse:
    """Invert scale*K + shift*M, with Dirichlet or natural FE boundaries."""
    def __init__(self, mass, stiffness, spacing, *, interior, scale=1., shift=1., workers=1):
        self.size = mass.shape[0]; self.interior = interior; self.workers = workers
        self.work = WorkCounter(); self.last_residual = 0.
        self.matrix = TensorSum(((shift, TensorProduct((mass,)*3)),)+tuple(
            (scale, TensorProduct(tuple(stiffness if j == i else mass for j in range(3)))) for i in range(3)))
        with self.work.measure('setup'):
            theta = np.pi*(np.arange(1, self.size+1)/(self.size+1) if interior else np.arange(self.size)/(self.size-1))
            mu = spacing*(2+np.cos(theta))/3
            kappa = 2*(1-np.cos(theta))/spacing
            self.denominator = shift*mu[:, None, None]*mu[None, :, None]*mu[None, None, :]
            for i in range(3):
                terms = [mu, mu, mu]; terms[i] = kappa
                self.denominator += scale*terms[0][:, None, None]*terms[1][None, :, None]*terms[2][None, None, :]
            self.boundary = np.ones(self.size)
            if not interior:
                self.boundary[[0, -1]] = np.sqrt(2.)

    def _inverse(self, rhs):
        shape = rhs.shape
        if rhs.ndim == 0 or shape[0] != self.size**3:
            raise ValueError('Tensor elliptic dimension mismatch')
        if not np.isfinite(rhs).all():
            raise ValueError('Nonfinite tensor elliptic RHS')
        if not np.any(rhs):
            return np.zeros_like(rhs)
        values = np.asarray(rhs).reshape((self.size,)*3+(-1,)).copy()
        transform = dstn if self.interior else dctn
        if not self.interior:
            for axis in range(3):
                weights = [1]*4; weights[axis] = self.size
                values *= self.boundary.reshape(weights)
        values = transform(values, type=1, axes=(0, 1, 2), norm='ortho', workers=self.workers)
        values /= self.denominator[..., None]
        values = transform(values, type=1, axes=(0, 1, 2), norm='ortho', workers=self.workers)
        if not self.interior:
            for axis in range(3):
                weights = [1]*4; weights[axis] = self.size
                values *= self.boundary.reshape(weights)
        return values.reshape(shape)

    @timed('solve')
    def solve(self, rhs):
        self.work.add('solves')
        result = self._inverse(np.asarray(rhs, dtype=float))
        norm = np.linalg.norm(rhs)
        self.last_residual = float(np.linalg.norm(self.matrix @ result.reshape(result.shape[0],-1)-rhs.reshape(rhs.shape[0],-1))/norm) if norm else 0.
        if not np.isfinite(self.last_residual) or self.last_residual > 1e-10:
            raise RuntimeError('Tensor elliptic inverse failed its true residual check')
        return result

    @timed('precondition')
    def relax(self, rhs):
        self.work.add('tensor_preconditions')
        return self._inverse(np.asarray(rhs, dtype=float))
