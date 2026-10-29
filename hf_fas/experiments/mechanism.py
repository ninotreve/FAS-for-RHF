"""Consistent-mass sine bands of the aligned horizontal orbital error."""
import numpy as np
from scipy.fft import dstn
from hf_fas.fem.scf import align

def spatial_error(space, error):
    """Exact consistent-Q1 mass energy in orthogonal tensor sine bands."""
    n = space.mesh.elements-1
    coefficients = dstn(error.reshape(n, n, n, -1), type=1, axes=(0, 1, 2), norm='ortho')
    k = np.arange(1, n+1)
    mass = space.mesh.spacing*(2+np.cos(np.pi*k/(n+1)))/3
    energy = np.sum(coefficients**2, axis=3)*mass[:, None, None]*mass[None, :, None]*mass[None, None, :]
    largest = np.maximum(np.maximum(k[:, None, None], k[None, :, None]), k[None, None, :])
    low = largest <= max(1, (n+1)//4)
    high = largest > (n+1)//2
    return dict(total=float(energy.sum()),low=float(energy[low].sum()),
                high=float(energy[high].sum()),middle=float(energy[~(low|high)].sum()))

def error_bands(model, state, reference):
    p = reference.orbitals[0]
    error = align(model, state.orbitals[0], p)-p
    error -= p @ (p.T @ (model.space.mass @ error))
    return spatial_error(model.space, error)
