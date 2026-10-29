"""Tensor bounds agree with the original bound without a 3D allocation."""
import numpy as np
import pytest
from hf_fas.tensor import TensorMesh, TensorQ1Space


@pytest.mark.parametrize('elements', [2, 3, 5, 8])
def test_tensor_mass_bounds_match_old_formula_and_bound_spectrum(elements, monkeypatch):
    space = TensorQ1Space(TensorMesh(elements, 4.))
    diagonal = space.mass.diagonal()
    expected = (diagonal.min()/8, 27*diagonal.max()/8)
    eigenvalues = np.linalg.eigvalsh(space.mass_1d.toarray()[1:-1, 1:-1])
    monkeypatch.setattr(space.mass, 'diagonal', lambda: pytest.fail('Allocated full 3D diagonal'))
    bounds = space.mass_spectral_bounds
    np.testing.assert_allclose(bounds, expected, rtol=2e-15)
    assert bounds[0] <= eigenvalues[0]**3
    assert bounds[1] >= eigenvalues[-1]**3
