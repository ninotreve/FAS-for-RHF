"""Direct Q1 block assembly from batched tensor potential stencils.

Only mesh geometry is cached. Every call evaluates the current Hartree,
exchange and multiplier fields; no stale Fock matrix survives a GS sweep.
"""
import numpy as np
from scipy.sparse import bsr_matrix


def _block_stencil(space):
    if space._block_stencil is None:
        pattern = space.mass.tocsr()
        pattern.sort_indices()
        cursor = pattern.indptr[:-1].copy()
        slots = []
        for operator, _, row, col in space.potential_stencil():
            positions = cursor[row].copy()
            if not np.array_equal(pattern.indices[positions], col):
                raise RuntimeError('Q1 neighbour order disagrees with mass sparsity')
            cursor[row] += 1
            slots.append((operator, row, col, positions))
        if not np.array_equal(cursor, pattern.indptr[1:]):
            raise RuntimeError('Incomplete Q1 block stencil')
        space._block_stencil = pattern, slots
    return space._block_stencil


def orbital_block_matrix(space, hcore, hartree, pairs, multipliers, *, batch_size=8):
    """The same vertex-major matrix as generic FE assembly, without kron.

    Exact symmetry permits upper-triangle packing. Near-symmetric supplied
    fields retain both entries rather than silently symmetrizing input.
    Temporary tensor batches are bounded independently of occupied count.
    """
    count = multipliers.shape[0]
    if count < 1 or batch_size < 1:
        raise ValueError('A nonempty orbital block and positive batch size are required')
    pattern, slots = _block_stencil(space)
    core = hcore.tocsr()
    core.sort_indices()
    same_pattern = (np.array_equal(core.indptr, pattern.indptr)
                    and np.array_equal(core.indices, pattern.indices))
    core_data = core.data if same_pattern else np.empty_like(pattern.data)
    if not same_pattern:
        for _, row, col, positions in slots:
            core_data[positions] = np.asarray(core[row, col]).ravel()
    data = np.empty((len(pattern.data), count, count))
    diagonal = np.arange(count)
    # Chunk the broadcast to avoid another full block-matrix temporary.
    for start in range(0, len(data), 16384):
        stop = start+16384
        data[start:stop] = -pattern.data[start:stop, None, None]*multipliers.T
        data[start:stop, diagonal, diagonal] += core_data[start:stop, None]
    symmetric = np.array_equal(pairs, pairs.swapaxes(1, 2))
    indices = list(zip(*np.triu_indices(count))) if symmetric else list(np.ndindex(count, count))
    fields = [None]+indices
    for start in range(0, len(fields), batch_size):
        selected = fields[start:start+batch_size]
        values = np.column_stack([hartree if ij is None else pairs[:, ij[0], ij[1]] for ij in selected])
        for operator, row, _, positions in slots:
            integrals = (operator @ values)[row]
            for column, ij in enumerate(selected):
                if ij is None:
                    data[positions[:, None], diagonal, diagonal] += integrals[:, column, None]
                else:
                    i, j = ij
                    data[positions, i, j] -= integrals[:, column]
                    if symmetric and i != j:
                        data[positions, j, i] -= integrals[:, column]
    return bsr_matrix((data, pattern.indices, pattern.indptr),
                      shape=(space.n_orbital*count,)*2, copy=False)
