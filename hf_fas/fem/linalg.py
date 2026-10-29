"""Reusable mass and auxiliary inverses, with family-invariant AMG stopping."""
from dataclasses import dataclass
import numpy as np
from scipy.sparse.linalg import splu
from .records import WorkCounter, timed


@dataclass(frozen=True)
class LinearSolveConfig:
    method: str = 'direct'
    tolerance: float = 1e-12
    max_iterations: int = 300


class SPDInverse:
    def __init__(self, matrix, config=None):
        self.matrix = matrix.tocsr()
        self.config = config or LinearSolveConfig()
        if (self.config.method not in ('direct', 'amg')
            or not np.isfinite(self.config.tolerance) or self.config.tolerance <= 0
            or not isinstance(self.config.max_iterations, int) or self.config.max_iterations < 1):
            raise ValueError('Invalid linear solve configuration')
        self.last_residual = 0.
        self.work = WorkCounter()
        with self.work.measure('setup'):
            if self.config.method == 'direct':
                self.factor = splu(matrix.tocsc())
            else:
                try:
                    import pyamg
                except ImportError as exc:
                    raise ImportError('Install hf-fas[fem] for the AMG backend') from exc
                self.amg = pyamg.smoothed_aggregation_solver(matrix, symmetry='hermitian')
                self.preconditioner = self.amg.aspreconditioner(cycle='V')

    @timed('solve')
    def solve(self, rhs):
        self.work.add('solves')
        rhs = np.asarray(rhs, dtype=float)
        if rhs.ndim == 0 or rhs.shape[0] != self.matrix.shape[0]:
            raise ValueError('Linear solve dimension mismatch')
        if not np.isfinite(rhs).all():
            raise ValueError('Nonfinite linear solve RHS')
        shape = rhs.shape
        b = rhs.reshape(shape[0], -1)
        if b.shape[1] == 0 or not np.any(b):
            self.last_residual = 0.
            return np.zeros_like(rhs)
        if self.config.method == 'direct':
            x = self.factor.solve(b)
        else:
            # CG on I_family tensor A. Common Frobenius scalars and one family
            # stop preserve orthogonal mixing of the RHS family, unlike
            # independently stopped per-pair solves.
            x = np.zeros_like(b); r = b.copy()
            z = self.preconditioner @ r; p = z.copy()
            rz = float(np.sum(r*z))
            for _ in range(self.config.max_iterations):
                self.work.add('cg_iterations')
                ap = self.matrix @ p
                denominator = float(np.sum(p*ap))
                if denominator <= 0 or not np.isfinite(denominator):
                    raise RuntimeError('AMG-preconditioned family CG lost positive curvature')
                alpha = rz/denominator; x += alpha*p; r -= alpha*ap
                if np.linalg.norm(r) <= self.config.tolerance*np.linalg.norm(b):
                    break
                z = self.preconditioner @ r; new = float(np.sum(r*z))
                p = z+(new/rz)*p; rz = new
            else:
                raise RuntimeError('AMG family solve reached its iteration cap')
        self.last_residual = float(np.linalg.norm(self.matrix @ x-b)/np.linalg.norm(b))
        if not np.isfinite(self.last_residual) or self.last_residual > max(20*self.config.tolerance, 1e-10):
            raise RuntimeError('Linear inverse failed its independently recomputed residual')
        return x.reshape(shape)

    @timed('precondition')
    def relax(self, rhs):
        if self.config.method == 'direct':
            return self.solve(rhs)
        shape = rhs.shape
        self.work.add('amg_cycles')
        return (self.preconditioner @ rhs.reshape(shape[0], -1)).reshape(shape)
