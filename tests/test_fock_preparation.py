"""A fixed DIIS density is constructed once for all trial Fock applications."""
import numpy as np
import pytest
from hf_fas.fem import MolecularFEMModel, Molecule, regression_molecules
from hf_fas.fem.scf import prepare_fock_action
from hf_fas.tensor import TensorMesh, TensorQ1Space, TensorCoulomb


@pytest.mark.parametrize('name', ['LiH', 'empty_beta'])
def test_prepared_fock_matches_explicit_density_mixture_and_freezes_weights(name):
    space = TensorQ1Space(TensorMesh(4, 4.))
    molecule = regression_molecules()['LiH'] if name == 'LiH' else Molecule.from_atoms([('H', (0, 0, 0))], spin=1)
    model = MolecularFEMModel(space, molecule, TensorCoulomb(space))
    rng = np.random.default_rng(420)
    first = model.initial_state()
    second = model.rebuild(tuple(p+.01*rng.normal(size=p.shape) for p in first.orbitals))
    states = (first, second); weights = np.array([1.2, -.2])
    for spin in range(len(first.orbitals)):
        action = prepare_fock_action(model, states, weights, spin)
        setups = model.work.values['mixed_fock_setups']
        for shape in [(space.n_orbital,), (space.n_orbital, 3)]:
            v = rng.normal(size=shape)
            matrix = v[:, None] if v.ndim == 1 else v
            expected = np.zeros_like(matrix)
            for w, state in zip(weights, states):
                occupied = state.orbitals[spin]
                field = model.hcore @ matrix+space.multiply_load(model.hartree(state.pairs), matrix)
                if occupied.shape[1]:
                    field -= space.source_adjoint(occupied, model.coulomb.integral_action(space.source_products(occupied, matrix)))
                expected += w*field
            np.testing.assert_allclose(action(v), expected[:, 0] if v.ndim == 1 else expected, atol=3e-13)
        assert model.work.values['mixed_fock_setups'] == setups
        probe = rng.normal(size=(space.n_orbital, 2))
        before = action(probe)
        weights[:] = [0., 1.]
        np.testing.assert_array_equal(action(probe), before)
        weights[:] = [1.2, -.2]
