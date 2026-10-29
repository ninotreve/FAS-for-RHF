"""FFT geometry for recorded work; historical integrated-kernel fallback only."""
import math
from scipy.fft import next_fast_len


def padded_shape(level):
    """Read actual runtime shape without changing the Coulomb fingerprint."""
    recorded = level.get('fft_padded_shape')
    metadata = level['coulomb']
    method = metadata.get('config', {}).get('method', 'fft')
    if recorded is None:
        if method != 'fft' or 'VGF' in metadata.get('backend', ''):
            raise ValueError('VGF work requires recorded actual fft_padded_shape')
        recorded = (next_fast_len(2*metadata['voxel_side']-1),)*3
    shape = tuple(recorded)
    if len(shape) != 3 or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in shape):
        raise ValueError('Invalid FFT padded shape')
    if method == 'vgf' and shape != (2*metadata['voxel_side'],)*3:
        raise ValueError('Recorded VGF FFT geometry differs from runtime policy')
    return shape


def fft_points(level):
    return math.prod(padded_shape(level))


def fft_weight(level):
    points = fft_points(level)
    return points*math.log2(points)


import numpy as np

def work_summary(models, work):
    weights = [float(np.prod(m.coulomb.padded)) for m in models]
    weights = [p * np.log2(p) for p in weights]
    levels = []
    for level, (counts, weight) in enumerate(zip(work, weights)):
        model = counts['model']
        rhs = counts['coulomb'].get('evaluated_source_rhs', 0)
        levels.append(dict(level=level, packed_scalar_rhs=rhs,
            relative_fft_weight=weight / weights[-1], fft_work=rhs * weight / weights[-1],
            response_calls=model.get('response_applications', 0),
            retracted_Newton_calls=model.get('time.Newton_retracted_step.calls', 0)))
    return dict(levels=levels, fft_work=sum(r['fft_work'] for r in levels),
                scope='All-level packed scalar RHS times padded P log2(P), normalized to finest; not total FLOPs')
