"""Horizontal Newton against a dense quotient oracle and independent endpoints."""
import numpy as np
import pytest
from scipy.linalg import null_space
from hf_fas.fem import (MolecularFEMModel, SCFConfig, regression_molecules,
    solve_molecular_scf, solve_molecular_quotient, Molecule)
from hf_fas.fem.coarse import CoarseConfig
from hf_fas.fem.quotient import HorizontalCoordinates, linear_horizontal_step
from hf_fas.tensor import TensorMesh, TensorQ1Space, TensorCoulomb


def model_for(name):
    space = TensorQ1Space(TensorMesh(4, 4.))
    return MolecularFEMModel(space, regression_molecules()[name], TensorCoulomb(space))


def test_linear_direction_matches_dense_quotient_and_rotates_covariantly():
    model = model_for('LiH'); state = model.initial_state()
    cfg = CoarseConfig(krylov_tolerance=1e-9)
    direction, report = linear_horizontal_step(model, state, cfg)
    assert report['converged'] and report['normal_direction_defect'] < 1e-12
    c = HorizontalCoordinates(model, state)
    # An independent explicit basis for the horizontal subspace.
    P = np.column_stack([c.project(v) for v in np.eye(c.size)])
    Q = null_space(np.eye(c.size)-P)
    A = np.column_stack([c.response(v) for v in Q.T])
    rhs = -c.project(c.coordinates.load_coordinates(model.space.mass @ model.residual(state).orbitals[0]).ravel())
    expected = Q @ np.linalg.solve(Q.T @ A, Q.T @ rhs)
    np.testing.assert_allclose(c.white(direction), expected, atol=2e-8, rtol=2e-8)
    q = np.array([[.8, -.6], [.6, .8]])
    rotated, other = linear_horizontal_step(model, state.rotate((q,)), cfg)
    assert other['converged']
    np.testing.assert_allclose(rotated, direction @ q, atol=2e-8, rtol=2e-8)


def test_projected_response_matches_rebuilt_gradient_difference():
    model = model_for('LiH'); state = model.initial_state()
    c = HorizontalCoordinates(model, state)
    rng = np.random.default_rng(12)
    v = c.project(rng.normal(size=c.size)); v /= np.linalg.norm(v)
    direction = c.physical(v)
    gradients = []
    step = 1e-5
    for sign in (-1, 1):
        trial = model.rebuild((state.orbitals[0]+sign*step*direction,))
        load = model.space.mass @ model.residual(trial).orbitals[0]
        gradients.append(c.project(c.coordinates.load_coordinates(load).ravel()))
    np.testing.assert_allclose(c.response(v), (gradients[1]-gradients[0])/(2*step),
                               rtol=2e-7, atol=2e-9)


@pytest.mark.parametrize('name', ['He', 'LiH'])
def test_complete_solver_matches_independent_SCF_endpoint(name):
    from hf_fas.experiments.fem_paper import endpoint_check
    model = model_for(name); initial = model.initial_state()
    scf = solve_molecular_scf(model, initial, SCFConfig(eigensolver='dense', tolerance=1e-10))
    result = solve_molecular_quotient(model, initial, CoarseConfig(tolerance=1e-10))
    assert result.converged, result.history[-1]
    assert scf.converged and endpoint_check(model, result.state, scf.state)['qualified']
    assert model.residual(result.state).norm(model.space) < 1e-10
    for row in result.history[:-1]:
        assert row['step'] > 0
        assert row['trials'][-1]['residual'] < row['residual']


def test_scope_and_iteration_budget_are_explicit():
    model = model_for('LiH'); state = model.initial_state()
    forced = model.rebuild((state.orbitals[0]*1.01,), retract=False)
    with pytest.raises(ValueError, match='unit Gram'):
        solve_molecular_quotient(model, forced)
    other = MolecularFEMModel(model.space, Molecule.from_atoms([('H', (0, 0, 0))], spin=1), model.coulomb)
    with pytest.raises(ValueError, match='closed-shell RHF'):
        solve_molecular_quotient(other)
    result = solve_molecular_quotient(model, state, CoarseConfig(max_iterations=1))
    assert not result.converged and len(result.history) <= 2
    _, report = linear_horizontal_step(model, state, CoarseConfig(krylov_iterations=1, krylov_restart=1))
    assert not report['converged'] and report['iterations'] == 1
