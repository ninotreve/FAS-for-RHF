"""Q1/VGF experiment construction and independent endpoint checks."""
from copy import copy
from dataclasses import asdict
import hashlib
import time
import numpy as np
from scipy.special import erf, erfc
from hf_fas.fem import FEMConfig, MolecularFEMModel, MolecularState, projector_distance
from hf_fas.fem.records import WorkCounter, model_work, work_difference
from hf_fas.tensor import TensorMesh, TensorQ1Space, TensorCoulomb, TensorCoulombConfig

def load_state(path, model):
    with np.load(path) as data:
        groups = [tuple((data[f'{prefix}_{i}'].copy() for i in range(len(model.molecule.occupations)))) for prefix in ('phi', 'pair', 'lambda')]
    state = MolecularState(*groups)
    model.validate(state)
    return state

def density(model, state):
    return sum((w * np.trace(model.space.sources(p), axis1=1, axis2=2) for (w, p) in zip(model.molecule.weights, state.orbitals)))

def endpoint_check(model, state, reference):
    difference = density(model, state) - density(model, reference)
    norm = lambda q: np.sqrt(max(0.0, q @ (model.space.pair_mass @ q)))
    de = abs(model.energy(state).electronic - model.energy(reference).electronic)
    dp = max((projector_distance(model.space, a, b) for (a, b) in zip(state.orbitals, reference.orbitals)))
    drho = float(norm(difference) / max(norm(density(model, reference)), 1e-30))
    residual = model.residual(state).norm(model.space)
    dne = abs(model.electron_count(state) - model.molecule.electrons)
    return dict(qualified=residual < 1e-08 and de < 1e-07 and (dp < 1e-06) and (drho < 1e-06) and (dne < 1e-09), residual=residual, energy_difference=de, projector_distance=dp, relative_density=drho, electron_count_error=dne)

def precision_endpoint_check(model, state):
    """Separate validation; share immutable geometry, never a solver state cache.

    Disable any assembled matrix so that the requested stricter FMM precision
    actually reaches the point engine. Fresh counters isolate validation work.
    """
    strict = copy(model)
    strict.work = WorkCounter()
    coulomb = model.coulomb.stricter()
    precision = dict(kernel_order=coulomb.config.kernel_order, kernel_tolerance=coulomb.config.kernel_tolerance, subdivisions=coulomb.config.subdivisions, fmm_tolerance=None)
    precision.update(method=coulomb.config.method)
    if coulomb.config.method == 'vgf':
        precision.update(kernel_order=None, vgf_precompute=coulomb.config.vgf_precompute, kernel_control='DCT_vs_streamed_inverse_FFT_at_fixed_voxel_partition')
    strict.coulomb = coulomb
    before = model_work(strict)
    raw = strict.residual(state).block_norms(model.space)
    rebuilt = strict.rebuild(state.orbitals, retract=False)
    restored = strict.residual(rebuilt).block_norms(model.space)
    high_energy = strict.energy(state).electronic
    energy_difference = abs(high_energy - model.energy(state).electronic)
    return dict(qualified=raw['total'] < 1e-08 and restored['total'] < 1e-08 and (energy_difference < 1e-09), **precision, raw_residual=raw, rebuilt_residual=restored, electronic_energy=high_energy, energy_difference=energy_difference, coulomb_backend=coulomb.fingerprint, work=work_difference(strict, before), scope='Independent endpoint validation, excluded from application timing')

def tensor_coulomb_config(args):
    return TensorCoulombConfig(method=getattr(args, 'coulomb_method', 'vgf'), vgf_precompute=getattr(args, 'vgf_precompute', 'dct'), subdivisions=getattr(args, 'voxel_subdivisions', 2), kernel_order=getattr(args, 'kernel_order', 256), workers=getattr(args, 'fft_workers', 1))

def build_tensor(args):
    if getattr(args, 'warp', 0.0) != 0.0:
        raise ValueError('Tensor backend requires an unwarped uniform box')
    started = time.perf_counter()
    mesh = TensorMesh(args.elements, args.half_width)
    meshes = [mesh]
    for _ in range(args.refinements):
        meshes.append(meshes[-1].uniform_refine())
    cfg = FEMConfig(mass_method='tensor', auxiliary_method='tensor', nuclear_order=args.nuclear_order)
    spaces = [TensorQ1Space(mesh, cfg) for mesh in meshes]
    ccfg = tensor_coulomb_config(args)
    maps = [TensorCoulomb(space, ccfg, kernel_cache=getattr(args, 'kernel_cache', None)) for space in spaces]
    levels = [dict(quality=s.mesh.quality(), orbital_dofs=s.n_orbital, pair_dofs=s.n_pair, coulomb=c.metadata, fft_padded_shape=list(c.padded), audit=c.audit(), kernel_cache=c.cache_info, setup_work=dict(coulomb=c.work.snapshot(), mass=s.mass_inverse.work.snapshot(), pair_mass=s.pair_mass_inverse.work.snapshot(), auxiliary=s.auxiliary_inverse.work.snapshot(), kinetic=s.kinetic_inverse.work.snapshot())) for (s, c) in zip(spaces, maps)]
    return (spaces, maps, dict(seconds=time.perf_counter() - started, backend='tensor_Q1_FFT', levels=levels))

def models_for(name, spaces, maps):
    from hf_fas.fem.molecule import paper_molecules
    return [MolecularFEMModel(s, paper_molecules()[name], c) for (s, c) in zip(spaces, maps)]

def gaussian_control(space, coulomb, half_width):
    source = space.project_function( lambda x: np.exp(-np.sum(x * x, axis=-1)) / np.pi ** 1.5)

    def potential(x):
        r = np.linalg.norm(x, axis=-1)
        return np.divide(erf(r), r, out=np.full_like(r, 2 / np.sqrt(np.pi)), where=r > 0)
    reference = space.project_function( potential)
    computed = coulomb(source)
    error = computed - reference
    norm = lambda x: np.sqrt(max(0.0, x @ (space.pair_mass @ x)))
    probes = np.array([[0.0, 0.0, 0.0], [0.5 * half_width, 0.0, 0.0], [0.0, 0.5 * half_width, 0.0]])
    point_error = coulomb.evaluate(source, probes) - potential(probes)
    tail = erfc(half_width)
    outside = 3 * tail - 3 * tail ** 2 + tail ** 3
    return dict(relative_projected_potential_error=float(norm(error) / norm(reference)), maximum_probe_error=float(np.max(abs(point_error))), bilinear=float(source @ coulomb.integral_action(source)), exact_whole_space_bilinear=float(np.sqrt(2 / np.pi)), represented_charge=float(source @ (space.pair_mass @ np.ones(space.n_pair))), exact_outside_source_mass=float(outside), outside_potential_bound_at_probes=float(2 * outside / half_width), scope='Projected FE potential uses the selected FFT quadrature; point probes independently integrate the voxel source directly. Includes source/voxel errors; whole-space tail bound stated separately')

def variational_qualification(model, state=None):
    state = model.initial_state() if state is None else state
    rng = np.random.default_rng(47)
    e = tuple((rng.normal(size=p.shape) * 0.01 for p in state.orbitals))
    eps = 1e-05
    a = model.rebuild(tuple((p + eps * d for (p, d) in zip(state.orbitals, e))), retract=False)
    b = model.rebuild(tuple((p - eps * d for (p, d) in zip(state.orbitals, e))), retract=False)
    derivative = (model.energy(a).electronic - model.energy(b).electronic) / (2 * eps)
    exact = sum((2 * w * np.sum(d * h) for (w, d, h) in zip(model.molecule.weights, e, model.action_load(state.orbitals, state.pairs))))
    response = model.response_load(state, e)
    fd = [(ha - hb) / (2 * eps) - model.space.mass @ (d @ lam) for (ha, hb, d, lam) in zip(model.action_load(a.orbitals, a.pairs), model.action_load(b.orbitals, b.pairs), e, state.multipliers)]
    response_error = max((np.linalg.norm(x - y) / max(1.0, np.linalg.norm(y)) for (x, y) in zip(response, fd)))
    derivative_error = abs(derivative - exact) / max(1.0, abs(exact))
    return dict(qualified=derivative_error < 1e-06 and response_error < 1e-06, derivative_relative_error=derivative_error, response_relative_error=response_error)
