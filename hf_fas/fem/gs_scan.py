"""Sequential BSR block GS: optional compilation, unchanged vertex ordering.

HF_FAS_GS_BACKEND=python selects the reference implementation; numba requires
the optional accelerator, and auto (default) uses it when installed. No
parallel updates or fast-math transformations are used. A numerically rank-deficient
block hands the unmodified current row back to the reference least-squares path.
"""
import os
import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


_LOCAL_RCOND = 1e-12


def backend():
    requested = os.environ.get('HF_FAS_GS_BACKEND', 'auto')
    if requested not in ('auto', 'python', 'numba'):
        raise ValueError('HF_FAS_GS_BACKEND must be auto, python or numba')
    if requested == 'numba' and njit is None:
        raise RuntimeError('Compiled GS requires numba; install hf-fas[performance]')
    return 'numba' if requested != 'python' and njit is not None else 'python'


def _reference(data, indices, indptr, rhs, x, weight, start_row=0):
    for row in range(start_row, len(x)):
        start, end = indptr[row:row+2]
        columns, blocks = indices[start:end], data[start:end]
        diagonal = np.flatnonzero(columns == row)
        if len(diagonal) != 1:
            raise RuntimeError('Orbital block matrix has no unique point diagonal')
        local = blocks[diagonal[0]]
        defect = rhs[row]-np.einsum('bij,bj->i', blocks, x[columns])
        singular = np.linalg.svd(local, full_matrices=False)[1]
        if singular[-1] > _LOCAL_RCOND*singular[0]:
            step = np.linalg.solve(local, defect)
        else:
            step = np.linalg.lstsq(local, defect, rcond=_LOCAL_RCOND)[0]
        x[row] += weight*step


def _compiled_impl(data, indices, indptr, rhs, x, weight):
    count = x.shape[1]
    defect = np.empty(count, dtype=np.float64)
    for row in range(len(x)):
        diagonal = -1
        # Same block/column summation order as the reference einsum.
        for i in range(count):
            value = 0.
            for entry in range(indptr[row], indptr[row+1]):
                for j in range(count):
                    value += data[entry, i, j]*x[indices[entry], j]
            defect[i] = rhs[row, i]-value
        for entry in range(indptr[row], indptr[row+1]):
            if indices[entry] == row:
                if diagonal != -1:
                    raise RuntimeError('Orbital block matrix has no unique point diagonal')
                diagonal = entry
        if diagonal == -1:
            raise RuntimeError('Orbital block matrix has no unique point diagonal')
        singular = np.linalg.svd(data[diagonal], full_matrices=False)[1]
        if singular[-1] <= _LOCAL_RCOND*singular[0]:
            return row+1
        step = np.linalg.solve(data[diagonal], defect)
        for i in range(count):
            x[row, i] += weight*step[i]
    return 0


_compiled = njit(cache=True)(_compiled_impl) if njit is not None else None


def scan(matrix, rhs, x, weight):
    """Update x in place using the existing lexicographic block GS sweep."""
    selected = backend()
    if selected == 'numba':
        failed = _compiled(matrix.data, matrix.indices, matrix.indptr, rhs, x, weight)
        if failed:
            _reference(matrix.data, matrix.indices, matrix.indptr, rhs, x, weight, failed-1)
    else:
        _reference(matrix.data, matrix.indices, matrix.indptr, rhs, x, weight)


def warmup():
    """Compile outside solver timing; exercise both SciPy BSR index widths."""
    if backend() == 'numba':
        for dtype in (np.int32, np.int64):
            _compiled(np.eye(2)[None].copy(), np.array([0], dtype=dtype),
                      np.array([0, 1], dtype=dtype), np.ones((1, 2)),
                      np.zeros((1, 2)), 1.)
