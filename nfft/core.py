from __future__ import division

import numpy as np
from scipy.sparse import csr_matrix

from .kernels import KERNELS
from .utils import nfft_matrix, fourier_sum, inv_fourier_sum


def ndft(x, f_hat):
    """Compute the nonuniform DFT

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO

    See Also
    --------
    infft : inverse nonuniform FFT
    """
    x, f_hat = map(np.asarray, (x, f_hat))
    assert x.ndim == 1
    assert f_hat.ndim == 1

    N = len(f_hat)
    assert N % 2 == 0

    k = -(N // 2) + np.arange(N)
    return np.dot(f_hat, np.exp(-2j * np.pi * x * k[:, None]))


def ndft_adjoint(x, f, N):
    """Compute the adjoint of the nonuniform DFT

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO

    See Also
    --------
    infft : inverse nonuniform FFT
    """
    x, f = np.broadcast_arrays(x, f)
    assert x.ndim == 1

    N = int(N)
    assert N % 2 == 0

    k = -(N // 2) + np.arange(N)
    return np.dot(f, np.exp(2j * np.pi * k * x[:, None]))


def nfft(x, f_hat, sigma=5, tol=1E-8, m=None, kernel='gaussian',
         use_fft=True, truncated=True):
    # Validate inputs
    x, f_hat = map(np.asarray, (x, f_hat))
    assert x.ndim == 1
    assert f_hat.ndim == 1

    N = len(f_hat)
    assert N % 2 == 0

    sigma = int(sigma)
    assert sigma >= 2

    n = N * sigma

    kernel = KERNELS.get(kernel, kernel)

    if m is None:
        m = kernel.estimate_m(tol, N, sigma)
    assert m <= n // 2

    k = -(N // 2) + np.arange(N)

    # Compute the NFFT
    ghat = f_hat / kernel.phi_hat(k, n, m, sigma) / n
    g = fourier_sum(ghat, N, n, use_fft=use_fft)
    mat = nfft_matrix(x, n, m, sigma, kernel, truncated=truncated)
    f = mat.dot(g)

    return f


def nfft_adjoint(x, f, N, sigma=5, tol=1E-8, m=None, kernel='gaussian',
                 use_fft=True, truncated=True):
    """Compute the inverse nonuniform FFT

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO

    See Also
    --------
    indft : inverse nonuniform DFT
    """
    # Validate inputs
    x, f = np.broadcast_arrays(x, f)
    assert x.ndim == 1

    N = int(N)
    assert N % 2 == 0

    sigma = int(sigma)
    assert sigma >= 2

    n = N * sigma

    kernel = KERNELS.get(kernel, kernel)

    if m is None:
        m = kernel.estimate_m(tol, N, sigma)
    assert m <= n // 2

    k = -(N // 2) + np.arange(N)

    # Compute the adjoint NFFT
    mat = nfft_matrix(x, n, m, sigma, kernel, truncated=truncated)
    g = mat.T.dot(f)
    ghat = inv_fourier_sum(g, N, n, use_fft=use_fft)
    fhat = ghat / kernel.phi_hat(k, n, m, sigma) / n

    return fhat
