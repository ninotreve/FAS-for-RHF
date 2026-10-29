"""Q1 orbitals/pairs, exact L2 product projection and its weak adjoints."""
from itertools import product
import os
import numpy as np
from scipy.sparse import coo_matrix, diags

from hf_fas.fem.space import FEMConfig
from .linalg import TensorProduct, TensorSum, TensorMassInverse, TensorEllipticInverse


def interpolation(axis, points):
    h = axis[1]-axis[0]
    left = np.clip(np.floor((points-axis[0])/h).astype(int), 0, len(axis)-2)
    fraction = (points-axis[left])/h
    return coo_matrix((np.column_stack((1-fraction, fraction)).ravel(),
        (np.repeat(np.arange(len(points)), 2), np.column_stack((left, left+1)).ravel())),
        shape=(len(points), len(axis))).tocsr()


def one_dimensional_factors(elements, spacing):
    count = elements+1
    diagonal = np.full(count, 2*spacing/3); diagonal[[0, -1]] /= 2
    mass = diags((np.full(count-1, spacing/6), diagonal, np.full(count-1, spacing/6)), (-1, 0, 1), format='csr')
    diagonal = np.full(count, 2/spacing); diagonal[[0, -1]] /= 2
    stiffness = diags((np.full(count-1, -1/spacing), diagonal, np.full(count-1, -1/spacing)), (-1, 0, 1), format='csr')
    return mass, stiffness


def hex_basis(points):
    corners = np.array(list(product((0, 1), repeat=3)))
    return np.prod(np.where(corners == 1, points[..., None, :], 1-points[..., None, :]), axis=-1)


def singular_nuclear_block(lower, h, center, order):
    """Signed face pyramids; four-point radial Gauss integrates degree seven."""
    q, w = np.polynomial.legendre.leggauss(order); q, w = (q+1)/2, w/2
    uv = np.stack(np.meshgrid(q, q, indexing='ij'), axis=-1).reshape(-1, 2)
    face_weights = np.outer(w, w).ravel()*h*h
    radial, rw = np.polynomial.legendre.leggauss(4); radial, rw = (radial+1)/2, rw/2
    matrix = np.zeros((8, 8))
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        for side in (0, 1):
            face = np.empty((len(uv), 3)); face[:, other] = lower[other]+h*uv
            face[:, axis] = lower[axis]+side*h
            height = (2*side-1)*(face[0, axis]-center[axis])
            if height == 0:
                continue
            vector = face-center
            distance = np.linalg.norm(vector, axis=1)
            rays = center+radial[None, :, None]*vector[:, None]
            basis = hex_basis((rays-lower)/h)
            weights = (height*face_weights/distance)[:, None]*(rw*radial)[None]
            matrix += np.einsum('qr,qra,qrb->ab', weights, basis, basis)
    return (matrix+matrix.T)/2


class TensorQ1Space:
    backend = 'tensor_q1'

    def __init__(self, mesh, config=None):
        self.mesh = mesh
        self.config = config or FEMConfig(mass_method='tensor', auxiliary_method='tensor', nuclear_order=8)
        if self.config.mass_method != 'tensor' or self.config.auxiliary_method != 'tensor' or self.config.pair_shift <= 0:
            raise ValueError('Q1 tensor space requires tensor mass/elliptic inverses and a positive pair shift')
        m, h = mesh.elements, mesh.spacing
        self.n_orbital, self.n_pair = (m-1)**3, (m+1)**3
        self.mass_1d, self.stiffness_1d = one_dimensional_factors(m, h)
        mv, kv = self.mass_1d[1:-1, 1:-1], self.stiffness_1d[1:-1, 1:-1]
        self.mass = TensorProduct((mv,)*3)
        self.stiffness = TensorSum(tuple((1., TensorProduct(tuple(kv if j == i else mv for j in range(3)))) for i in range(3))).tocsr()
        self.pair_mass = TensorProduct((self.mass_1d,)*3)
        self.pair_stiffness = TensorSum(tuple((1., TensorProduct(tuple(self.stiffness_1d if j == i else self.mass_1d for j in range(3)))) for i in range(3)))
        self.auxiliary = TensorSum(self.pair_stiffness.terms+((self.config.pair_shift, self.pair_mass),))
        self.mass_inverse = TensorMassInverse(mv)
        self.mass_cholesky = self.mass_inverse.root
        self.pair_mass_inverse = TensorMassInverse(self.mass_1d)
        workers = max(1, int(os.getenv('OMP_NUM_THREADS', '1')))
        self.auxiliary_inverse = TensorEllipticInverse(self.mass_1d, self.stiffness_1d, h,
            interior=False, shift=self.config.pair_shift, workers=workers)
        self._kinetic_inverse = TensorEllipticInverse(mv, kv, h, interior=True, scale=.5, shift=1., workers=workers)
        self.evaluation = TensorProduct((interpolation(mesh.axis, mesh.axis)[:, 1:-1],)*3)
        q, _ = np.polynomial.legendre.leggauss(2)
        points = (mesh.axis[:-1, None]+h*(q+1)[None]/2).ravel()
        b = interpolation(mesh.axis, points)
        self.pair_quadrature = TensorProduct((b,)*3)
        self.orbital_quadrature = TensorProduct((b[:, 1:-1],)*3)
        self.quadrature_weight = (h/2)**3
        self._stencil = None
        self._block_stencil = None

    @property
    def kinetic_inverse(self):
        return self._kinetic_inverse

    @property
    def mass_spectral_bounds(self):
        # The orbital mass is the cube of the interior 1D mass. Avoid
        # materializing an O(n_V) Kronecker diagonal merely to take its extrema.
        diagonal = self.mass_1d.diagonal()[1:-1]
        return float(diagonal.min()**3/8), float(27*diagonal.max()**3/8)

    def make_coulomb(self):
        from .coulomb import TensorCoulomb
        return TensorCoulomb(self)

    def source_products(self, left, right):
        a = self.orbital_quadrature @ left
        b = a if left is right else self.orbital_quadrature @ right
        result = np.empty((self.n_pair, left.shape[1], right.shape[1]))
        symmetric = left is right
        indices = list(zip(*np.triu_indices(left.shape[1]))) if symmetric else list(product(range(left.shape[1]), range(right.shape[1])))
        for start in range(0, len(indices), 4):
            selected = indices[start:start+4]
            values = np.column_stack([a[:, i]*b[:, j] for i, j in selected])
            projected = self.pair_mass_inverse.solve(self.pair_quadrature.T @ (self.quadrature_weight*values))
            for k, (i, j) in enumerate(selected):
                result[:, i, j] = projected[:, k]
                if symmetric:
                    result[:, j, i] = projected[:, k]
        return result

    def sources(self, orbitals):
        return self.source_products(orbitals, orbitals)

    def source_adjoint(self, occupied, pair_loads):
        count, trials = pair_loads.shape[1:]
        if count == 0:
            return np.zeros((self.n_orbital, trials))
        fields = self.pair_mass_inverse.solve(pair_loads.reshape(self.n_pair, -1))
        fields = (self.pair_quadrature @ fields).reshape(-1, count, trials)
        product_load = np.einsum('qi,qij->qj', self.orbital_quadrature @ occupied, fields)
        return self.orbital_quadrature.T @ (self.quadrature_weight*product_load)

    def multiply_load(self, field, orbitals):
        potential = self.pair_quadrature @ field
        sampled = self.orbital_quadrature @ orbitals
        values = potential[:, None]*sampled if sampled.ndim == 2 else potential*sampled
        return self.orbital_quadrature.T @ (self.quadrature_weight*values)

    def exchange_load(self, pairs, orbitals):
        if not orbitals.shape[1]:
            return np.zeros_like(orbitals)
        fields = (self.pair_quadrature @ pairs.reshape(self.n_pair, -1)).reshape((-1,)+pairs.shape[1:])
        values = np.einsum('qij,qj->qi', fields, self.orbital_quadrature @ orbitals)
        return self.orbital_quadrature.T @ (self.quadrature_weight*values)

    def potential_stencil(self):
        """Exact trilinear potential integrals for each of the 27 neighbours."""
        m, h = self.mesh.elements, self.mesh.spacing
        n = m-1
        if self._stencil is None:
            index = np.arange(n**3).reshape((n,)*3)
            stencil = []
            for delta in product((-1, 0, 1), repeat=3):
                row_slice = tuple(slice(max(0, -d), min(n, n-d)) for d in delta)
                col_slice = tuple(slice(max(0, d), min(n, n+d)) for d in delta)
                factors = []
                for d in delta:
                    row = np.arange(n)
                    if d == 0:
                        columns = np.column_stack((row, row+1, row+2)); values = np.tile(h*np.array([1/12, 1/2, 1/12]), (n, 1))
                    else:
                        columns = np.column_stack((row+1+min(d, 0), row+2+min(d, 0))); values = np.full((n, 2), h/12)
                    factors.append(coo_matrix((values.ravel(), (np.repeat(row, columns.shape[1]), columns.ravel())), shape=(n, m+1)).tocsr())
                stencil.append((TensorProduct(factors), row_slice, index[row_slice].ravel(), index[col_slice].ravel()))
            self._stencil = stencil
        return self._stencil

    def potential_matrix(self, field):
        n = self.mesh.elements-1
        rows, cols, values = [], [], []
        for operator, selection, row, col in self.potential_stencil():
            rows.append(row); cols.append(col)
            values.append((operator @ field).reshape((n,)*3)[selection].ravel())
        return coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(self.n_orbital,)*2).tocsr()

    def orbital_block_matrix(self, hcore, hartree, pairs, multipliers):
        from .relaxation import orbital_block_matrix
        return orbital_block_matrix(self, hcore, hartree, pairs, multipliers)

    def pair_scale(self, values):
        return self.pair_mass_inverse.solve(self.auxiliary @ values.reshape(self.n_pair, -1)).reshape(values.shape)

    def forcing_correction(self, values):
        return self.auxiliary_inverse.solve(self.pair_mass @ values.reshape(self.n_pair, -1)).reshape(values.shape)

    def interpolation_to(self, fine, degree):
        """Shared transfer selector: 1 is orbital, 2 is pair; both spaces are Q1."""
        if fine.mesh.half_width != self.mesh.half_width:
            raise ValueError('Tensor transfers require the same physical box')
        b = interpolation(self.mesh.axis, fine.mesh.axis if degree == 2 else fine.mesh.axis[1:-1])
        return TensorProduct((b if degree == 2 else b[:, 1:-1],)*3)

    def project_function(self, function, order=8):
        q, w = np.polynomial.legendre.leggauss(order)
        axis = (self.mesh.axis[:-1, None]+self.mesh.spacing*(q+1)[None]/2).ravel()
        b = interpolation(self.mesh.axis, axis)
        evaluation = TensorProduct((b,)*3)
        points = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1)
        weight = np.tile(w*self.mesh.spacing/2, self.mesh.elements)
        values = function(points)*weight[:, None, None]*weight[None, :, None]*weight[None, None, :]
        return self.pair_mass_inverse.solve(evaluation.T @ values.ravel())

    def nuclear_matrix(self, centers, charges):
        m, h = self.mesh.elements, self.mesh.spacing
        q, w = np.polynomial.legendre.leggauss(max(3, self.config.nuclear_order)); q, w = (q+1)/2, w/2
        sample = np.stack(np.meshgrid(q, q, q, indexing='ij'), axis=-1).reshape(-1, 3)
        weight = (w[:, None, None]*w[None, :, None]*w[None, None, :]).ravel()*h**3
        basis = hex_basis(sample)
        indices = np.stack(np.meshgrid(*[np.arange(m)]*3, indexing='ij'), axis=-1).reshape(-1, 3)
        corners = np.array(list(product((0, 1), repeat=3)))
        rows, cols, data = [], [], []
        for first in range(0, len(indices), 128):
            ids = indices[first:first+128]; lower = -self.mesh.half_width+h*ids
            points = lower[:, None]+h*sample[None]
            local = np.zeros((len(ids), 8, 8))
            for center, charge in zip(centers, charges):
                near = np.linalg.norm(lower+h/2-center, axis=1) <= 2*np.sqrt(3)*h
                if np.any(~near):
                    potential = -charge/np.linalg.norm(points[~near]-center, axis=-1)
                    local[~near] += np.einsum('eq,q,qa,qb->eab', potential, weight, basis, basis, optimize=True)
                for k in np.flatnonzero(near):
                    local[k] -= charge*singular_nuclear_block(lower[k], h, center, self.config.nuclear_order)
            nodes = ids[:, None]+corners[None]
            interior = np.all((nodes > 0) & (nodes < m), axis=-1)
            reduced = np.ravel_multi_index(np.clip(nodes-1, 0, m-2).transpose(2, 0, 1), (m-1,)*3)
            selected = interior[:, :, None] & interior[:, None, :]
            rows.append(np.broadcast_to(reduced[:, :, None], local.shape)[selected])
            cols.append(np.broadcast_to(reduced[:, None, :], local.shape)[selected])
            data.append(local[selected])
        return coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(self.n_orbital,)*2).tocsr()
