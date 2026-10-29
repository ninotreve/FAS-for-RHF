"""Compiled sequential GS must retain the reference block update semantics."""
import numpy as np
import pytest
from scipy.sparse import bsr_matrix
from hf_fas.fem import gs_scan

pytestmark = pytest.mark.skipif(gs_scan.njit is None, reason='optional numba unavailable')


@pytest.mark.parametrize('count', [1, 2, 3, 5, 6, 7])
@pytest.mark.parametrize('weight', [1., .125])
def test_compiled_scan_matches_sequential_reference(count, weight, monkeypatch):
    rng = np.random.default_rng(82+count)
    blocks = rng.normal(size=(7, count, count))
    for k in (0, 3, 6):
        blocks[k] += 4*np.eye(count)
    # Unsorted neighbors, and an invertible local block requiring pivoting.
    if count > 1:
        blocks[3, :2, :2] = [[0., 2.], [3., 1.]]
    matrix = bsr_matrix((blocks, np.array([0, 1, 2, 1, 0, 1, 2]),
                         np.array([0, 2, 5, 7])), shape=(3*count, 3*count))
    rhs = rng.normal(size=(3, count)); initial = rng.normal(size=rhs.shape)
    expected = initial.copy(); actual = initial.copy()
    monkeypatch.setenv('HF_FAS_GS_BACKEND', 'python')
    gs_scan.scan(matrix, rhs, expected, weight)
    monkeypatch.setenv('HF_FAS_GS_BACKEND', 'numba')
    gs_scan.scan(matrix, rhs, actual, weight)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_singular_block_resumes_at_failed_row_with_lstsq(monkeypatch):
    data = np.array([[[2., 1.], [1., 3.]], [[1., 0.], [0., 1.]],
                     [[1., 2.], [2., 4.]], [[2., 0.], [0., 2.]], np.eye(2)])
    matrix = bsr_matrix((data, [0, 0, 1, 1, 2], [0, 1, 3, 5]), shape=(6, 6))
    rhs = np.arange(6.).reshape(3, 2); actual = np.ones((3, 2)); expected = actual.copy()
    monkeypatch.setenv('HF_FAS_GS_BACKEND', 'python')
    gs_scan.scan(matrix, rhs, expected, .5)
    monkeypatch.setenv('HF_FAS_GS_BACKEND', 'numba')
    gs_scan.scan(matrix, rhs, actual, .5)
    np.testing.assert_allclose(actual, expected, atol=2e-14)


@pytest.mark.parametrize('columns,ptr', [([1, 0], [0, 1, 2]), ([0, 0, 1], [0, 2, 3])])
@pytest.mark.parametrize('backend', ['python', 'numba'])
def test_nonunique_or_missing_diagonal_rejected(columns, ptr, backend, monkeypatch):
    monkeypatch.setenv('HF_FAS_GS_BACKEND', backend)
    matrix = bsr_matrix((np.ones((len(columns), 1, 1)), columns, ptr), shape=(2, 2))
    with pytest.raises(RuntimeError, match='unique point diagonal'):
        gs_scan.scan(matrix, np.ones((2, 1)), np.zeros((2, 1)), 1.)


def test_full_forced_lih_trial_equivalence(monkeypatch):
    from hf_fas.tensor import TensorMesh, TensorQ1Space, TensorCoulomb
    from hf_fas.fem import MolecularFEMModel, MolecularResidual, regression_molecules
    from hf_fas.fem.relaxation import block_sweep
    from hf_fas.experiments.fem_run import state_arrays
    space = TensorQ1Space(TensorMesh(4, 4.))
    model = MolecularFEMModel(space, regression_molecules()['LiH'], TensorCoulomb(space))
    state = model.initial_state(); forcing = MolecularResidual.zeros(state)
    forcing.orbitals[0][:] = .001; forcing.pairs[0][:] = .0001
    forcing.grams[0][:] = [[.1, .02], [.02, -.05]]
    state = model.rebuild(state.orbitals, forcing)
    monkeypatch.setenv('HF_FAS_GS_BACKEND', 'python')
    expected = block_sweep(model, state, forcing, .5)
    monkeypatch.setenv('HF_FAS_GS_BACKEND', 'numba')
    actual = block_sweep(model, state, forcing, .5)
    for name, value in state_arrays(expected).items():
        np.testing.assert_allclose(state_arrays(actual)[name], value, rtol=3e-12, atol=3e-12)
    assert model.residual(actual, forcing).block_norms(space)['pair'] < 1e-11


def test_two_complete_lih_cycles_match_reference(monkeypatch):
    from hf_fas.tensor import TensorMesh, TensorQ1Space, TensorCoulomb
    from hf_fas.fem import MolecularFEMModel, MolecularResidual, regression_molecules, FEMHierarchy, FASConfig
    from hf_fas.experiments.fem_run import state_arrays
    models = [MolecularFEMModel(s, regression_molecules()['LiH'], TensorCoulomb(s))
              for s in [TensorQ1Space(TensorMesh(n, 4.)) for n in (4, 8)]]
    start = models[-1].initial_state(); final = []
    for selected in ('python', 'numba'):
        monkeypatch.setenv('HF_FAS_GS_BACKEND', selected)
        hierarchy = FEMHierarchy(models); state = start.copy()
        for _ in range(2):
            state = hierarchy.cycle(1, state, MolecularResidual.zeros(state), FASConfig())
        final.append(state_arrays(state))
    for name in final[0]:
        np.testing.assert_allclose(final[0][name], final[1][name], rtol=3e-11, atol=3e-11)


def test_screening_panel_is_closed_shell_and_regression_defaults_unchanged():
    from hf_fas.fem.molecule import paper_molecules, regression_molecules
    assert set(regression_molecules()) == {'He', 'LiH', 'BH', 'H2O'}
    for name, electrons in [('LiH', 4), ('BeHplus', 4), ('Li2', 6), ('BeH2', 6)]:
        m = paper_molecules()[name]
        assert m.electrons == electrons and m.occupations == (electrons//2,)
        assert m.restricted and m.spin == 0
