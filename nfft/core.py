from __future__ import division

import numpy as np

from .kernels import KERNELS
from .utils import nfft_matrix, fourier_sum, inv_fourier_sum


def ndft(x, f_hat):
    """Compute the non-equispaced direct Fourier transform

    f_j = \sum_{-N/2 \le k < N/2} \hat{f}_k \exp(-2 \pi i k x_j)

    Parameters
    ----------
    x : array_like, shape=(M,)
        The locations of the data points.
    f_hat : array_like, shape=(N,)
        The amplitudes at each wave number k = range(-N/2, N/2)

    Returns
    -------
    f : ndarray, shape=(M,)
        The direct Fourier summation corresponding to x

    See Also
    --------
    nfft : non-equispaced fast Fourier transform
    ndft_adjoint : adjoint non-equispaced direct Fourier transform
    nfft_adjoint : adjoint non-equispaced fast Fourier transform
    """
    x, f_hat = map(np.asarray, (x, f_hat))
    assert x.ndim == 1
    assert f_hat.ndim == 1

    N = len(f_hat)
    assert N % 2 == 0

    k = -(N // 2) + np.arange(N)
    return np.dot(f_hat, np.exp(-2j * np.pi * x * k[:, None]))


def ndft_adjoint(x, f, N):
    """Compute the adjoint non-equispaced direct Fourier transform

    \hat{f}_k = \sum_{0 \le j < N} f_j \exp(2 \pi i k x_j)

    where k = range(-N/2, N/2)

    Parameters
    ----------
    x : array_like, shape=(M,)
        The locations of the data points.
    f : array_like, shape=(M,)
        The amplitudes at each location x
    N : int
        The number of frequencies at which to evaluate the result

    Returns
    -------
    f_hat : ndarray, shape=(N,)
        The amplitudes corresponding to each wave number k = range(-N/2, N/2)

    See Also
    --------
    nfft_adjoint : adjoint non-equispaced fast Fourier transform
    ndft : non-equispaced direct Fourier transform
    nfft : non-equispaced fast Fourier transform
    """
    x, f = np.broadcast_arrays(x, f)
    assert x.ndim == 1

    N = int(N)
    assert N % 2 == 0

    k = -(N // 2) + np.arange(N)
    return np.dot(f, np.exp(2j * np.pi * k * x[:, None]))


def nfft(x, f_hat, sigma=3, tol=1E-8, m=None, kernel='gaussian',
         use_fft=True, truncated=True):
    """Compute the non-equispaced fast Fourier transform

    f_j = \sum_{-N/2 \le k < N/2} \hat{f}_k \exp(-2 \pi i k x_j)

    Parameters
    ----------
    x : array_like, shape=(M,)
        The locations of the data points. Each value in x should lie
        in the range [-1/2, 1/2).
    f_hat : array_like, shape=(N,)
        The amplitudes at each wave number k = range(-N/2, N/2).
    sigma : int (optional, default=5)
        The oversampling factor for the FFT gridding.
    tol : float (optional, default=1E-8)
        The desired tolerance of the truncation approximation.
    m : int (optional)
        The half-width of the truncated window. If not specified, ``m`` will
        be estimated based on ``tol``.
    kernel : string or NFFTKernel (optional, default='gaussian')
        The desired convolution kernel for the calculation.
    use_fft : bool (optional, default=True)
        If True, use the FFT rather than DFT for fast computation.
    truncated : bool (optional, default=True)
        If True, use a fast truncated approximate summation matrix.
        If False, use a slow full summation matrix.

    Returns
    -------
    f : ndarray, shape=(M,)
        The approximate Fourier summation evaluated at points x

    See Also
    --------
    ndft : non-equispaced direct Fourier transform
    nfft_adjoint : adjoint non-equispaced fast Fourier transform
    ndft_adjoint : adjoint non-equispaced direct Fourier transform
    """
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
    m = int(m)
    assert m <= n // 2

    k = -(N // 2) + np.arange(N)

    # Compute the NFFT
    ghat = f_hat / kernel.phi_hat(k, n, m, sigma) / n
    g = fourier_sum(ghat, N, n, use_fft=use_fft)
    mat = nfft_matrix(x, n, m, sigma, kernel, truncated=truncated)
    f = mat.dot(g)

    return f


def nfft_adjoint(x, f, N, sigma=3, tol=1E-8, m=None, kernel='gaussian',
                 use_fft=True, truncated=True):
    """Compute the adjoint non-equispaced fast Fourier transform

    \hat{f}_k = \sum_{0 \le j < N} f_j \exp(2 \pi i k x_j)

    where k = range(-N/2, N/2)

    Parameters
    ----------
    x : array_like, shape=(M,)
        The locations of the data points.
    f : array_like, shape=(M,)
        The amplitudes at each location x
    N : int
        The number of frequencies at which to evaluate the result
    sigma : int (optional, default=5)
        The oversampling factor for the FFT gridding.
    tol : float (optional, default=1E-8)
        The desired tolerance of the truncation approximation.
    m : int (optional)
        The half-width of the truncated window. If not specified, ``m`` will
        be estimated based on ``tol``.
    kernel : string or NFFTKernel (optional, default='gaussian')
        The desired convolution kernel for the calculation.
    use_fft : bool (optional, default=True)
        If True, use the FFT rather than DFT for fast computation.
    truncated : bool (optional, default=True)
        If True, use a fast truncated approximate summation matrix.
        If False, use a slow full summation matrix.

    Returns
    -------
    f_hat : ndarray, shape=(N,)
        The approximate amplitudes corresponding to each wave number
        k = range(-N/2, N/2)

    See Also
    --------
    ndft_adjoint : adjoint non-equispaced direct Fourier transform
    nfft : non-equispaced fast Fourier transform
    ndft : non-equispaced direct Fourier transform
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
    m = int(m)
    assert m <= n // 2

    k = -(N // 2) + np.arange(N)

    # Compute the adjoint NFFT
    mat = nfft_matrix(x, n, m, sigma, kernel, truncated=truncated)
    g = mat.T.dot(f)
    ghat = inv_fourier_sum(g, N, n, use_fft=use_fft)
    fhat = ghat / kernel.phi_hat(k, n, m, sigma) / n

    return fhat
