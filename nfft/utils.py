from __future__ import division

import numpy as np

from scipy import fftpack, sparse

from .kernels import KERNELS, NFFTKernel


def shifted(x):
    """Shift x values to the range [-0.5, 0.5)"""
    return -0.5 + (x + 0.5) % 1


def nfft_matrix(x, n, m, sigma, kernel, truncated):
    """Compute the nfft matrix

    This is the matrix that encodes the (truncated) convolution that projects
    the irregularly-sampled data onto a regular grid.

    Parameters
    ----------
    x : ndarray, shape=(M, D)
        the array of coordinates in the range [-1/2, 1/2)
    n : ndarray, shape=(D,)
        the size of the oversampled frequency grid in each dimension
    m : ndarray, shape=(D,)
        the half-width of the truncated convolution window in each dimension.
        Only referenced if truncated is True.
    sigma : int
        oversampling factor
    kernel : string or NFFTKernel object
        the object providing the kernel interface
    truncated : boolean
        if True, then return the sparse, truncated matrix based on ``m``.
        if False, then return the full convolution matrix.

    Returns
    -------
    mat : ndarray or csr_matrix
        The [M, prod(n)] nfft matrix. If truncated is True, then the result
        is a sparse CSR matrix representing the truncated convolution with
        M * prod(m) nonzero components. If truncated is False, the result is
        a dense array.
    """
    kernel = KERNELS.get(kernel, kernel)
    assert isinstance(kernel, NFFTKernel)

    x = np.atleast_1d(x)
    if x.ndim == 1:
        x = x[:, None]
    n = np.atleast_1d(n)
    m = np.atleast_1d(m)

    if truncated:
        grid = np.stack(np.meshgrid(*(np.arange(-mi, mi) for mi in m), indexing='ij'))[:, None].T
        col_ind = grid + np.floor(n * x).astype(int)
        diff = (x - col_ind / n)
        vals = kernel.phi(shifted(diff.reshape(-1, len(m))), n, m, sigma).reshape(diff.shape[:-1])
        strides = np.cumprod(n[::-1])
        strides = strides[-1] / strides[::-1]
        col_ind_flat = np.dot((col_ind + n // 2) % n, strides)
        indices = np.rollaxis(col_ind_flat, -1).ravel()
        data = np.rollaxis(vals, -1).ravel()
        indptr = np.prod(2 * m) * np.arange(x.shape[0] + 1)
        mat = sparse.csr_matrix((data, indices, indptr), shape=(x.shape[0], np.prod(n)))
    else:
        grids = np.meshgrid(*(np.linspace(-0.5, 0.5, ni, endpoint=False) for ni in n))
        x_grid = np.array(grids).reshape(len(n), -1).T
        assert x_grid.shape == (np.prod(n), len(n))

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
