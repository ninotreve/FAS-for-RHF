"""Free-space 1/r VGF weights with compact real-even precomputation.

Vico--Greengard--Ferrando (2016), doi:10.1016/j.jcp.2016.07.028;
Mayani et al., arXiv:2405.02603, section 2.4 (DCT compression).
"""
import numpy as np
from scipy.fft import dctn, irfft, rfftn


def vgf_kernel(n, spacing, *, workers=1, precompute='dct'):
    """Return 2N-grid convolution weights, including source volume h^3.

    Sample the analytic 1/r transform on the independent (2N+1)^3
    octant of the 4N frequency grid. DCT-I and division by (4N)^3
    reproduce its inverse DFT, including zero/Nyquist endpoints.
    ``fft`` independently computes the same weights with streamed inverse
    real FFTs; it never allocates a full (4N)^3 array and is used for
    fixed-discretization endpoint validation.
    """
    if (not isinstance(n, (int, np.integer)) or n < 1 or not np.isfinite(spacing)
            or spacing <= 0 or workers < 1 or precompute not in ('dct','fft')):
        raise ValueError('Invalid VGF grid or precomputation policy')
    p, count = 4*n, 2*n+1
    radius = np.sqrt(3)*n*spacing
    k = 2*np.pi*np.arange(count)/(p*spacing)
    norm = k[:, None, None]**2+k[None, :, None]**2+k[None, None, :]**2
    np.sqrt(norm, out=norm)
    norm *= radius/(2*np.pi)
    weights = np.sinc(norm)
    del norm
    weights **= 2
    weights *= 2*np.pi*radius**2
    if precompute == 'dct':
        weights = dctn(weights, type=1, norm='backward', overwrite_x=True, workers=workers)
        weights /= p**3
    else:
        # Each irfft includes its own 1/p normalization. Stream slices so
        # validation needs only the octant plus O(N^2) transform workspace.
        for axis in range(3):
            view = np.moveaxis(weights, axis, 0)
            for row in range(count):
                view[:, row, :] = irfft(view[:, row, :], n=p, axis=0, workers=workers)[:count]
        del view
    signed = np.arange(2*n)
    indices = np.abs(np.where(signed < n, signed, signed-2*n))
    cropped = weights[np.ix_(indices, indices, indices)]
    del weights
    # These displacement planes are unused on the original N-grid.
    cropped[n, :, :] = 0
    cropped[:, n, :] = 0
    cropped[:, :, n] = 0
    return rfftn(cropped, workers=workers)
