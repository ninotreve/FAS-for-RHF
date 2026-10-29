"""Free-space Q1 Coulomb actions: default DCT-VGF or integrated-voxel FFT.

Both use consistent mass, exact Q1-to-voxel means and their weak transpose.
``vgf`` uses symmetric midpoint VGF weights; ``fft`` retains the original
double-voxel Galerkin integrals. Both apply free-space convolution, without
periodic images or a Poisson boundary closure.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import scipy
from scipy.fft import rfftn, irfftn, next_fast_len
from scipy.integrate import quad
from scipy.special import erf, erfc, roots_legendre

from hf_fas.fem.records import WorkCounter, timed
from .space import interpolation
from .linalg import TensorProduct
from .vgf import vgf_kernel


@dataclass(frozen=True)
class TensorCoulombConfig:
    method: str = 'vgf'
    subdivisions: int = 2
    kernel_order: int = 256
    kernel_tolerance: float = 1e-10
    workers: int = 1
    assembly_batch: int = 4
    symmetric_pair_sources: bool = True
    vgf_precompute: str = 'dct'


@lru_cache(maxsize=16)
def _legendre_rule(order):
    return roots_legendre(order)


def gaussian_voxel_axis(times, offsets):
    """Integral of (1-|u|)*exp(-t^2*(d+u)^2), without small-t cancellation."""
    times = np.asarray(times).reshape(-1)
    offsets = np.asarray(offsets, dtype=float).reshape(-1)
    values = np.empty((len(times), len(offsets)))
    nodes, weights = _legendre_rule(16)
    u, w = (nodes+1)/2, weights/2
    for row, t in enumerate(times):
        if t <= 1.:
            values[row] = (np.exp(-(t*(offsets[:, None]+u))**2)
                +np.exp(-(t*(offsets[:, None]-u))**2)) @ (w*(1-u))
        else:
            d = offsets
            result = np.sqrt(np.pi)/(2*t)*(2*d*erfc(t*d)-(d+1)*erfc(t*(d+1))-(d-1)*erfc(t*(d-1)))
            result += (np.exp(-(t*(d+1))**2)-2*np.exp(-(t*d)**2)+np.exp(-(t*(d-1))**2))/(2*t*t)
            zero = d == 0
            result[zero] = np.sqrt(np.pi)*erf(t)/t+np.expm1(-t*t)/(t*t)
            values[row] = np.maximum(result, 0.)
    return values


def laplace_rule(order, scale=1.):
    nodes, weights = _legendre_rule(order)
    u, w = (nodes+1)/2, weights/2
    times = u/((1-u)*scale)
    return times, (2/np.sqrt(np.pi))*w/((1-u)**2*scale)


def voxel_kernel_samples(offsets, order=128):
    offsets = np.abs(np.asarray(offsets, dtype=int))
    times, weights = laplace_rule(order, np.sqrt(offsets.max()+1))
    axis = gaussian_voxel_axis(times, np.arange(offsets.max()+1))
    return np.sum(weights[:, None]*np.prod(axis[:, offsets], axis=2), axis=0)


def voxel_kernel_reference(offset):
    """Adaptive infinite-interval integration, independent of the fixed rule."""
    return _voxel_reference(tuple(np.abs(np.asarray(offset,dtype=int))))


@lru_cache(maxsize=128)
def _voxel_reference(offset):
    indices = np.asarray(offset)
    scale = max(1.,float(np.linalg.norm(indices)))
    def integrand(time):
        values = gaussian_voxel_axis([time/scale], indices)[0]
        return (2/np.sqrt(np.pi))*float(np.prod(values))
    return quad(integrand, 0., np.inf, epsabs=1e-13, epsrel=1e-12, limit=300)[0]/scale


def _array_digest(values):
    digest = hashlib.sha256()
    digest.update(str((values.shape, values.dtype.str)).encode())
    digest.update(memoryview(np.ascontiguousarray(values)).cast('B'))
    return digest.hexdigest()


def prism_integral(lower, upper):
    """Integral of 1/r over axis-aligned boxes, also when r=0 is inside."""
    result = np.zeros(lower.shape[:-1])
    for corner in np.ndindex((2, 2, 2)):
        xyz = np.stack([upper[..., i] if side else lower[..., i] for i, side in enumerate(corner)], axis=-1)
        radius = np.linalg.norm(xyz, axis=-1)
        value = np.zeros_like(radius)
        for i in range(3):
            x, y, z = xyz[..., i], xyz[..., (i+1)%3], xyz[..., (i+2)%3]
            logarithm = np.log(x+radius, out=np.zeros_like(radius), where=x+radius > 0)
            ratio = np.divide(y*z, x*radius, out=np.zeros_like(radius), where=x*radius != 0)
            value += y*z*logarithm-.5*x*x*np.arctan(ratio)
        result += (-1.)**(3-sum(corner))*value
    return result


class TensorCoulomb:
    def __init__(self, space, config=None, *, kernel_cache=None):
        self.space = space; self.config = config or TensorCoulombConfig()
        cfg = self.config
        if (cfg.method not in ('fft','vgf') or cfg.vgf_precompute not in ('dct','fft')
            or not isinstance(cfg.subdivisions, int) or cfg.subdivisions < 1
            or cfg.kernel_order < 16 or cfg.workers < 1 or cfg.assembly_batch < 1
            or not 0 < cfg.kernel_tolerance < 1):
            raise ValueError('Invalid tensor Coulomb settings')
        self.work = WorkCounter(); self.matrix = None; self.kernel_cache = kernel_cache
        self.side = space.mesh.elements*cfg.subdivisions
        self.voxel_spacing = space.mesh.spacing/cfg.subdivisions
        self.axis = -space.mesh.half_width+(np.arange(self.side)+.5)*self.voxel_spacing
        self.evaluation = TensorProduct((interpolation(space.mesh.axis, self.axis),)*3)
        self.padded = ((2*self.side if cfg.method=='vgf' else next_fast_len(2*self.side-1)),)*3
        self.kernel_fft, self.cache_info, self.kernel_check = self._cached_kernel()
        config_metadata = asdict(cfg)
        if cfg.method=='fft':
            config_metadata.pop('vgf_precompute')
        self.metadata = dict(mesh=space.mesh.fingerprint, config=config_metadata, space_config=asdict(space.config),
            backend='Q1_projected_source_voxel_Galerkin_FFT_v1', source_projection='exact_consistent_L2_Q1',
            boundary='free_space_zero_extended_source', orbital_boundary='homogeneous_Dirichlet',
            pair_boundary='unconstrained_Q1', self_interaction='double_voxel_integral',
            voxel_side=self.side, voxel_spacing=self.voxel_spacing, assembled=False)
        if cfg.method=='vgf':
            self.metadata.update(backend='Q1_projected_source_DCT_VGF_FFT_v1',
                self_interaction='VGF_Nystrom_weight', pair_form='symmetric_midpoint_weak_return',
                precompute=cfg.vgf_precompute,
                point_evaluation='direct_integrated_voxel_source_control')
        self.fingerprint = hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()

    def _build_kernel(self):
        with self.work.measure('kernel_setup'):
            if self.config.method=='vgf':
                kernel = vgf_kernel(self.side,self.voxel_spacing,workers=self.config.workers,
                                    precompute=self.config.vgf_precompute)
                kernel *= self.voxel_spacing**3  # point potential -> weak volume load
                check = dict(qualified=bool(np.isfinite(kernel).all()),
                    construction='analytic_truncated_1_over_r',precompute=self.config.vgf_precompute,
                    scope='Finite analytic construction; independent inverse-FFT endpoint check is separate; no spatial-accuracy certificate')
                if not check['qualified']:
                    raise RuntimeError('Nonfinite VGF kernel')
                return kernel,check
            times, weights = laplace_rule(self.config.kernel_order, np.sqrt(self.side))
            axis = gaussian_voxel_axis(times, np.arange(self.side))
            xy = (axis[:, :, None]*axis[:, None, :]).reshape(len(times), -1)
            octant = (xy.T @ (weights[:, None]*axis)).reshape((self.side,)*3)
            probes = np.array([[0,0,0], [1,0,0], [1,1,1], [self.side-1,0,0], [self.side-1]*3])
            computed = octant[tuple(probes.T)]
            reference = np.array([voxel_kernel_reference(offset) for offset in probes])
            relative = float(np.max(abs(computed-reference)/reference))
            check = dict(qualified=relative < self.config.kernel_tolerance, offsets=probes.tolist(),
                computed=computed.tolist(), adaptive_reference=reference.tolist(), maximum_relative_error=relative)
            if not check['qualified']:
                raise RuntimeError(f'FFT kernel quadrature failed independent accuracy check: {relative}')
            index = np.arange(self.padded[0]); offset = np.where(index < self.side, index, index-self.padded[0])
            supported = np.abs(offset) < self.side; safe = np.minimum(np.abs(offset), self.side-1)
            kernel = octant[np.ix_(safe, safe, safe)]*self.voxel_spacing**5
            kernel *= supported[:, None, None] & supported[None, :, None] & supported[None, None, :]
            return rfftn(kernel, workers=self.config.workers), check

    def _cached_kernel(self):
        identity = dict(format='integrated_voxel_fft_v1', side=self.side, spacing=self.voxel_spacing,
            order=self.config.kernel_order, tolerance=self.config.kernel_tolerance,
            implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            numpy=np.__version__, scipy=scipy.__version__)
        if self.config.method=='vgf':
            identity.update(format='DCT_VGF_midpoint_fft_v1',precompute=self.config.vgf_precompute,
                vgf_implementation=hashlib.sha256(Path(__file__).with_name('vgf.py').read_bytes()).hexdigest())
            identity.pop('order')
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if self.kernel_cache is None:
            kernel, check = self._build_kernel()
            return kernel, dict(status='disabled', key=key), check
        directory = Path(self.kernel_cache); directory.mkdir(parents=True, exist_ok=True)
        path = directory/(key+'.npz')
        with (directory/(key+'.lock')).open('a+b') as lock:
            with self.work.measure('kernel_cache_wait'):
                fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                with self.work.measure('kernel_cache_load'), np.load(path, allow_pickle=False) as data:
                    metadata = json.loads(str(data['metadata'].item())); kernel = data['kernel']
                if metadata['identity'] != identity or metadata['digest'] != _array_digest(kernel):
                    raise ValueError('Invalid FFT kernel-cache identity/checksum')
                check = metadata['check']; status = 'hit'
            else:
                kernel, check = self._build_kernel(); status = 'built'
                metadata = dict(identity=identity, digest=_array_digest(kernel), check=check)
                temporary = None
                try:
                    with self.work.measure('kernel_cache_store'), tempfile.NamedTemporaryFile(dir=directory, suffix='.npz', delete=False) as handle:
                        temporary = Path(handle.name)
                        np.savez(handle, kernel=kernel, metadata=json.dumps(metadata, sort_keys=True))
                    os.replace(temporary, path)
                finally:
                    if temporary is not None and temporary.exists():
                        temporary.unlink()
        expected = self.padded[:2]+(self.padded[2]//2+1,)
        if kernel.shape != expected or not np.isfinite(kernel).all() or not check['qualified']:
            raise ValueError('Invalid FFT kernel-cache contents')
        self.work.add('kernel_cache_hits' if status == 'hit' else 'kernel_cache_builds')
        return kernel, dict(status=status, key=key, path=str(path.resolve())), check

    def stricter(self):
        return TensorCoulomb(self.space, replace_config(self.config), kernel_cache=self.kernel_cache)

    @timed('voxel_projection_and_FFT')
    def _integral_action(self, sources):
        values = (self.evaluation @ sources).reshape((self.side,)*3+(-1,))
        self.work.add('fft_calls')
        with self.work.measure('FFT_convolution'):
            spectrum = rfftn(values, s=self.padded, axes=(0,1,2), workers=self.config.workers)
            spectrum *= self.kernel_fft[..., None]
            load = irfftn(spectrum, s=self.padded, axes=(0,1,2), workers=self.config.workers, overwrite_x=True)
            load = load[:self.side, :self.side, :self.side].reshape(self.side**3, -1)
        return self.evaluation.T @ load

    @timed('integral_action')
    def integral_action(self, sources):
        sources = np.asarray(sources, dtype=float)
        if sources.shape[0] != self.space.n_pair or not np.isfinite(sources).all():
            raise ValueError('Invalid Q1 source coefficients')
        self.work.add('integral_families')
        shape = sources.shape; flat = sources.reshape(self.space.n_pair, -1)
        self.work.add('source_rhs', flat.shape[1])
        if not np.any(flat):
            return np.zeros_like(sources)
        symmetric = self.config.symmetric_pair_sources and sources.ndim == 3 and shape[1] == shape[2] and np.array_equal(sources, sources.swapaxes(1,2))
        if symmetric:
            i,j = np.triu_indices(shape[1]); unique = sources[:,i,j]
            self.work.add('packed_pair_families')
        else:
            unique = flat
        self.work.add('evaluated_source_rhs', unique.shape[1])
        computed = np.empty_like(unique)
        for start in range(0, unique.shape[1], self.config.assembly_batch):
            end = start+self.config.assembly_batch
            computed[:,start:end] = self._integral_action(unique[:,start:end])
        if symmetric:
            result = np.empty(shape); result[:,i,j] = computed; result[:,j,i] = computed
            return result
        return computed.reshape(shape)

    @timed('projected_potential')
    def __call__(self, sources):
        self.work.add('projected_families')
        return self.space.pair_mass_inverse.solve(self.integral_action(sources))

    def evaluate(self, sources, targets):
        """Direct free-space potential of the voxel-averaged source.

        A physical point diagnostic, using exact voxel integrals independently
        of either FFT quadrature and of the projected Q1 output potential.
        """
        sources = np.asarray(sources); shape = sources.shape[1:]
        values = self.evaluation @ sources.reshape(self.space.n_pair, -1)
        targets = np.asarray(targets).reshape(-1,3)
        output = np.zeros((len(targets), values.shape[1]))
        indices = np.arange(self.side**3)
        for first in range(0, len(indices), 4096):
            selected = indices[first:first+4096]
            nodes = np.column_stack(np.unravel_index(selected, (self.side,)*3))
            centers = -self.space.mesh.half_width+(nodes+.5)*self.voxel_spacing
            lower = centers[:,None]-targets[None]-self.voxel_spacing/2
            upper = lower+self.voxel_spacing
            output += prism_integral(lower, upper).T @ values[selected]
        return output.reshape((len(targets),)+shape)

    def audit(self, seed=17, samples=4):
        rng = np.random.default_rng(seed); x,y = rng.normal(size=(2,self.space.n_pair,samples))
        gx,gy = self.integral_action(x), self.integral_action(y)
        scale = max(1.,np.linalg.norm(x)*np.linalg.norm(gy),np.linalg.norm(y)*np.linalg.norm(gx))
        return dict(reciprocity_relative=float(abs(np.sum(x*gy)-np.sum(y*gx))/scale),
            sampled_minimum_bilinear=float(np.min(np.sum(x*gx,axis=0))), kernel_check=self.kernel_check,
            scope='Kernel construction and finite algebraic probes; voxel/source/domain errors require refinement')


def replace_config(config):
    from dataclasses import replace
    if config.method=='vgf':
        alternative='fft' if config.vgf_precompute=='dct' else 'dct'
        return replace(config,vgf_precompute=alternative,kernel_tolerance=min(1e-12,config.kernel_tolerance))
    return replace(config, kernel_order=max(192, config.kernel_order+64), kernel_tolerance=min(1e-12,config.kernel_tolerance))
