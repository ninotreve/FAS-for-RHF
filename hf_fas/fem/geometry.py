"""Mass-metric frame checks and cancellation-free subspace comparisons."""
import numpy as np
from scipy.linalg import cholesky, solve_triangular


class InvalidTrialState(ValueError):
    """An inadmissible numerical candidate; a line search may reject it."""


def symmetric_trial_matrix(matrix):
    """Remove roundoff skew from a mathematically symmetric computed matrix."""
    if not np.isfinite(matrix).all():
        raise InvalidTrialState('Nonfinite computed trial matrix')
    scale=max(1.,float(np.max(abs(matrix)))) if matrix.size else 1.
    if matrix.size and np.linalg.norm(matrix-matrix.T,ord=np.inf)>128*np.finfo(float).eps*scale*len(matrix):
        raise InvalidTrialState('Computed trial matrix lost numerical symmetry')
    return .5*matrix+.5*matrix.T


def checked_gram(mass, frame, condition_limit=1e12):
    if not np.isfinite(frame).all():
        raise InvalidTrialState('Nonfinite trial frame')
    gram = symmetric_trial_matrix(frame.T @ (mass @ frame))
    if gram.size:
        values = np.linalg.eigvalsh((gram+gram.T)/2)
        if values[0] <= 0 or values[-1]/values[0] > condition_limit:
            raise InvalidTrialState('Rank-deficient or ill-conditioned trial frame')
    return gram


def orthonormal_frame(mass, frame):
    gram = checked_gram(mass, frame)
    if not gram.size:
        return frame.copy()
    lower = cholesky(gram, lower=True)
    return solve_triangular(lower, frame.T, lower=True).T


def projector_distance(space, left, right):
    """||P_left-P_right||_HS from complementary residuals, without N-N."""
    a = orthonormal_frame(space.mass, left)
    b = orthonormal_frame(space.mass, right)
    overlap = a.T @ (space.mass @ b)
    ea, eb = a-b @ overlap.T, b-a @ overlap
    squared = float(np.sum(ea*(space.mass @ ea))+np.sum(eb*(space.mass @ eb)))
    return float(np.sqrt(max(0., squared)))


def checked_alignment(left, right, mass):
    overlap = right.T @ (mass @ left)
    u, values, vt = np.linalg.svd(overlap)
    if len(values) and values[-1] <= np.finfo(float).eps*len(values)*values[0]:
        raise InvalidTrialState('Singular normalized occupied overlap')
    return u @ vt
