"""Fixed methane geometries and prescribed initial perturbations."""
import numpy as np

def methane_centers(variant='canonical'):
    hydrogen = 2.06/np.sqrt(3)*np.array([[1, 1, 1], [1, -1, -1],
                                         [-1, 1, -1], [-1, -1, 1]], dtype=float)
    centers = np.vstack([np.zeros(3), hydrogen])
    if variant == 'rotated':
        axis = np.array([1., 2., 3.])/np.sqrt(14)
        angle = np.pi/7
        x, y, z = axis
        cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        rotation = np.cos(angle)*np.eye(3)+(1-np.cos(angle))*np.outer(axis, axis)+np.sin(angle)*cross
        centers = centers @ rotation.T
    elif variant == 'distorted':
        centers[1] *= 1.03
        centers[2] += np.array([0., .03, -.02])
    elif variant != 'canonical':
        raise ValueError('Unknown methane geometry')
    return centers

def smooth_direction(model, state, seed):
    """Fixed low-sine random coefficients; no reference orbitals enter the start."""
    p = state.orbitals[0]
    cells = model.space.mesh.elements
    modes = np.arange(1, min(4, cells-1)+1)
    waves = np.sin(np.pi*np.outer(np.arange(1, cells)/cells, modes))
    coefficients = np.random.default_rng(seed).normal(size=(len(modes),)*3+(p.shape[1],))
    direction = np.einsum('ia,jb,kc,abcd->ijkd', waves, waves, waves, coefficients).reshape(p.shape)
    direction -= p @ np.linalg.solve(model.gram(p), p.T @ (model.space.mass @ direction))
    norm = np.sqrt(np.sum(direction*(model.space.mass @ direction)))
    if norm < 1e-12:
        raise ValueError('Prescribed perturbation has no horizontal component')
    return direction*np.sqrt(p.shape[1])/norm

def construct_start(model, continuation, variant):
    if variant in ('continuation', 'continuation-perturbed'):
        base = continuation.copy()
    elif variant in ('core', 'core-perturbed'):
        base = model.initial_state(seed=2026102303)
    else:
        raise ValueError('Unknown prescribed initial family')
    if variant.endswith('perturbed'):
        amplitude, seed = (.1, 2026102304) if variant.startswith('continuation') else (.3, 2026102305)
        direction = smooth_direction(model, base, seed)
        return model.rebuild((base.orbitals[0]+amplitude*direction,))
    return base
