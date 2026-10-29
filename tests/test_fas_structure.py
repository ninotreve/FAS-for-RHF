"""Forced FAS fixed points and occupied-rotation contracts, including safeguards."""
import numpy as np
import pytest
from scipy.sparse import bsr_matrix

from hf_fas.fem import FEMHierarchy, FASConfig, MolecularFEMModel, regression_molecules
from hf_fas.fem import gs_scan
from hf_fas.fem.coarse import CoarseConfig, solve_forced_coarse
from hf_fas.fem.geometry import InvalidTrialState, symmetric_trial_matrix
from hf_fas.fem.model import MolecularState, symmetric_power
from hf_fas.tensor import TensorCoulomb, TensorMesh, TensorQ1Space


@pytest.fixture(params=['python', 'numba'])
def gs_backend(request, monkeypatch):
    if request.param == 'numba' and gs_scan.njit is None:
        pytest.skip('optional numba unavailable')
    monkeypatch.setenv('HF_FAS_GS_BACKEND', request.param)
    return request.param


@pytest.mark.parametrize('shift', [0., 1e-14])
def test_rank_deficient_gs_block_rotates_without_unstable_direct_solve(gs_backend, shift):
    local = np.array([[1., 2.], [2., 4.]])+shift*np.eye(2)
    rhs = np.array([[1., 0.]])
    expected = np.linalg.lstsq(local, rhs[0], rcond=1e-12)[0]
    for angle in (0., .1, .3, .7):
        q = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        matrix = bsr_matrix(((q.T @ local @ q)[None], [0], [0, 1]), shape=(2, 2))
        actual = np.zeros((1, 2))
        gs_scan.scan(matrix, rhs @ q, actual, 1.)
        np.testing.assert_allclose(actual[0], expected @ q, rtol=2e-13, atol=2e-14)


def test_zero_gs_block_uses_zero_minimum_norm_step(gs_backend):
    matrix = bsr_matrix((np.zeros((1, 2, 2)), [0], [0, 1]), shape=(2, 2))
    actual = np.array([[.2, -.3]])
    gs_scan.scan(matrix, np.ones((1, 2)), actual, 1.)
    np.testing.assert_array_equal(actual, [[.2, -.3]])


@pytest.fixture(scope='module')
def models():
    result = []
    for elements in (3, 6):
        space = TensorQ1Space(TensorMesh(elements, 4.))
        result.append(MolecularFEMModel(space, regression_molecules()['LiH'], TensorCoulomb(space)))
    return result


def forced_state(model):
    """All three forcing blocks are active; auxiliaries are independently chosen."""
    rng = np.random.default_rng(821)
    initial = model.initial_state()
    p = initial.orbitals[0] @ np.array([[1.1, .2], [0., .8]])
    w = initial.pairs[0]+.01*rng.normal(size=initial.pairs[0].shape)
    w = (w+w.swapaxes(1, 2))/2
    state = MolecularState((p,), (w,), (np.array([[.3, .2], [.2, -.4]]),))
    return state, model.operator(state)


def assert_state_equal(actual, expected, *, exact=False):
    for name in ('orbitals', 'pairs', 'multipliers'):
        for a, b in zip(getattr(actual, name), getattr(expected, name)):
            if exact:
                np.testing.assert_array_equal(a, b)
            else:
                np.testing.assert_allclose(a, b, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize('rebuild_initial', [False, True])
@pytest.mark.parametrize('ill_conditioned', [False, True])
def test_exact_forced_coarse_state_returns_before_reconstruction(models, monkeypatch,
                                                                rebuild_initial, ill_conditioned):
    model = models[0]
    state, _ = forced_state(model)
    if ill_conditioned:
        state.orbitals = (model.initial_state().orbitals[0] @ np.diag([1e-4, 1.]),)
        assert np.linalg.cond(model.gram(state.orbitals[0])) == pytest.approx(1e8)
    forcing = model.operator(state)
    original = state.copy()
    assert model.residual(state, forcing).norm(model.space) == 0.
    def unexpected(*args, **kwargs):
        pytest.fail('Converged forced input must not be reconstructed or retracted')
    monkeypatch.setattr(model, 'rebuild', unexpected)
    monkeypatch.setattr(model, 'retract', unexpected)
    monkeypatch.setattr(model.space, 'forcing_correction', unexpected)
    callbacks = []
    result, report, residual = solve_forced_coarse(model, state, forcing,
        rebuild_initial=rebuild_initial, return_residual=True,
        callback=lambda x, row: callbacks.append(row))
    assert_state_equal(result, original, exact=True)
    assert_state_equal(state, original, exact=True)
    assert report['converged'] and report['final_residual'] == 0.
    assert len(report['history']) == len(callbacks) == 1
    assert residual.norm(model.space) == 0.


@pytest.mark.parametrize('correction', ['component', 'geometric'])
@pytest.mark.parametrize('sweeps', [0, 2])
def test_complete_cycle_preserves_every_exact_forced_state_block(models, correction, sweeps):
    model = models[-1]
    state, forcing = forced_state(model)
    original = state.copy()
    for block in (forcing.orbitals, forcing.pairs, forcing.grams):
        assert any(np.any(x) for x in block)
    hierarchy = FEMHierarchy(models)
    result = hierarchy.cycle(1, state, forcing,
        FASConfig(correction=correction, pre_sweeps=sweeps, post_sweeps=sweeps))
    assert_state_equal(result, original, exact=True)
    assert_state_equal(state, original, exact=True)
    assert model.residual(result, forcing).norm(model.space) == 0.


@pytest.mark.parametrize('correction', ['component', 'geometric'])
def test_complete_forced_cycle_rotates_covariantly(models, gs_backend, correction):
    model = models[-1]
    state, forcing = forced_state(model)
    rng = np.random.default_rng(822)
    forcing.orbitals = (forcing.orbitals[0]+.03*rng.normal(size=state.orbitals[0].shape),)
    q = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    config = FASConfig(correction=correction, pre_sweeps=1, post_sweeps=1,
                       coarse=CoarseConfig(max_iterations=2))
    hierarchy, rotated = FEMHierarchy(models), FEMHierarchy(models)
    result = hierarchy.cycle(1, state.copy(), forcing, config)
    other = rotated.cycle(1, state.rotate((q,)), forcing.rotate((q,)), config)
    # Exercise an actual nonlinear cycle and coarse Newton solve, not an early exit.
    assert model.residual(result, forcing).norm(model.space) < model.residual(state, forcing).norm(model.space)
    assert any('linear' in row for report in hierarchy.coarse_records for row in report['history'])
    assert_state_equal(other, result.rotate((q,)))
    assert hierarchy.cycle_records[0]['selected_step'] == rotated.cycle_records[0]['selected_step']


@pytest.mark.parametrize('accepted', [False, True])
def test_matrix_symmetry_admissibility_is_invariant(accepted):
    rng = np.random.default_rng(823)
    q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    skew = np.array([[0., 1., 2.], [-1., 0., 3.], [-2., -3., 0.]])
    for check, magnitude in ((symmetric_trial_matrix, 1e-15 if accepted else 1e-10),
                             (lambda a: symmetric_power(a, .5), 1e-13 if accepted else 1e-8)):
        matrix = np.diag([1., 2., 3.])+magnitude*skew
        if accepted:
            np.testing.assert_allclose(check(q.T @ matrix @ q), q.T @ check(matrix) @ q, atol=2e-13)
        else:
            for candidate in (matrix, q.T @ matrix @ q):
                with pytest.raises(ValueError):
                    check(candidate)


def test_roundoff_skew_near_old_infinity_norm_boundary_rotates():
    # Mixing two columns increases the old row-sum skew test by sqrt(2),
    # despite leaving the Frobenius skew norm and matrix scale unchanged.
    cutoff = 128*np.finfo(float).eps*3
    matrix = np.eye(3)
    matrix[0, 1], matrix[1, 0] = .45*cutoff, -.45*cutoff
    c = np.sqrt(.5)
    q = np.array([[1., 0., 0.], [0., c, -c], [0., c, c]])
    for candidate in (matrix, q.T @ matrix @ q):
        symmetric_trial_matrix(candidate)


def test_target_symmetry_near_old_entrywise_boundary_rotates():
    # This rank-two skew matrix has entries below the old tolerance;
    # rotation concentrates its skew vector into an entry above it.
    skew = np.array([[0., .45, .45], [-.45, 0., .45], [-.45, -.45, 0.]])*1e-11
    axis = np.array([1., -1., 1.])/np.sqrt(3.)
    first = np.array([1., 1., 0.])/np.sqrt(2.)
    q = np.column_stack((first, np.cross(axis, first), axis))
    matrix = np.eye(3)+skew
    for candidate in (matrix, q.T @ matrix @ q):
        symmetric_power(candidate, .5)


@pytest.mark.parametrize('error', [9e-11, 2e-10])
def test_gram_restoration_acceptance_uses_invariant_error(models, monkeypatch, error):
    model = models[0]
    frame = model.initial_state().orbitals[0]
    angle = np.pi/8
    q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    gram_error = np.diag([error, 0.])
    for rotation in (np.eye(2), q):
        # Inject the same Gram error in two bases, isolating the acceptance
        # check from the rounding errors of the square-root computation.
        measured = np.eye(2)+rotation.T @ gram_error @ rotation
        monkeypatch.setattr(model, 'gram', lambda p: measured)
        if error < 1e-10:
            model.retract(frame @ rotation, np.eye(2))
        else:
            with pytest.raises(InvalidTrialState, match='target check'):
                model.retract(frame @ rotation, np.eye(2))


@pytest.mark.parametrize('block', ['pairs', 'multipliers'])
@pytest.mark.parametrize('accepted', [False, True])
def test_state_symmetry_admissibility_is_invariant(models, block, accepted):
    model = models[0]
    state = model.initial_state()
    perturbation = (1e-13 if accepted else 1e-8)*np.array([[0., 1.], [-1., 0.]])
    getattr(state, block)[0][...] += perturbation
    angle = .3
    q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    for candidate in (state, state.rotate((q,))):
        if accepted:
            model.validate(candidate)
        else:
            with pytest.raises(ValueError, match='symmetric'):
                model.validate(candidate)
