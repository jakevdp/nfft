from __future__ import division

import numpy as np

from scipy import fftpack, sparse


def shifted(x):
    """Shift x values to the range [-0.5, 0.5)"""
    return -0.5 + (x + 0.5) % 1


def nfft_matrix(x, n, m, sigma, kernel, truncated):
    """Compute the nfft matrix

    This is the matrix that encodes the (truncated) convolution that projects
    the irregularly-sampled data onto a regular grid.

    Parameters
    ----------
    x : ndarray, shape=M
        the array of coordinates in the range [-1/2, 1/2)
    n : int
        the size of the oversampled frequency grid
    m : int
        the half-width of the truncated convolution window.
        Only referenced if truncated is True.
    sigma : int
        oversampling factor
    kernel : NFFTKernel object
        the object providing the kernel interface
    truncated : boolean
        if True, then return the sparse, truncated matrix based on ``m``.
        if False, then return the full convolution matrix.

    Returns
    -------
    mat : ndarray or csr_matrix
        The [len(x), n] nfft matrix. If truncated is True, then the result
        is a sparse CSR matrix representing the truncated convolution.
        If truncated is False, the result is a dense array.
    """
    if truncated:
        col_ind = np.floor(n * x[:, np.newaxis]).astype(int) + np.arange(-m, m)
        val = kernel.phi(shifted(x[:, None] - col_ind / n), n, m, sigma)
        col_ind = (col_ind + n // 2) % n
        indptr = np.arange(len(x) + 1) * col_ind.shape[1]
        mat = sparse.csr_matrix((val.ravel(), col_ind.ravel(), indptr),
                                shape=(len(x), n))
    else:
        x_grid = np.linspace(-0.5, 0.5, n, endpoint=False)
        mat = kernel.phi(shifted(x_grid - x[:, None]), n, m, sigma)

    return mat


def fourier_sum(ghat, N, n, use_fft=True):
    """Evaluate the Fourier transform at N <= n points"""
    assert len(ghat) == N
    assert n >= N
    assert N % 2 == n % 2 == 0
    if use_fft:
        ghat_n = np.concatenate([ghat[N // 2:],
                                 np.zeros(n - N, dtype=ghat.dtype),
                                 ghat[:N // 2]])
        g = fftpack.fftshift(fftpack.fft(ghat_n))
    else:
        k = -(N // 2) + np.arange(N)
        x_grid = np.linspace(-0.5, 0.5, n, endpoint=False)[:, None]
        g = np.exp(-2j * np.pi * k * x_grid).dot(ghat)
    return g


def inv_fourier_sum(g, N, n, use_fft=True):
    """Evaluate the inverse Fourier transform at N <= n points"""
    assert len(g) == n
    assert n >= N
    assert N % 2 == n % 2 == 0
    if use_fft:
        ghat_n = fftpack.ifft(fftpack.fftshift(g))
        ghat = n * np.concatenate([ghat_n[-N // 2:], ghat_n[:N // 2]])
    else:
        k = -(N // 2) + np.arange(N)[:, None]
        x_grid = np.linspace(-0.5, 0.5, n, endpoint=False)
        ghat = np.exp(2j * np.pi * k * x_grid).dot(g)
    return ghat
