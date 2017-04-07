import numpy as np

from ..utils import nfft_matrix, fourier_sum, inv_fourier_sum
from ..kernels import KERNELS

from numpy.testing import assert_allclose
import pytest

kernel_types = sorted(KERNELS.keys())


@pytest.mark.parametrize('N', [50, 100, 200])
@pytest.mark.parametrize('sigma', [2, 3, 4])
@pytest.mark.parametrize('m', [10, 20, 30])
@pytest.mark.parametrize('kernel', kernel_types)
def test_nfft_matrix_shape_nnz(N, sigma, m, kernel, rseed=0):
    # Test that the shape and number of nonzero entries is as expected
    rand = np.random.RandomState(rseed)
    x = -0.5 + rand.rand(N)
    n = sigma * N
    kernel = KERNELS[kernel]

    mat = nfft_matrix(x, n, m, sigma, kernel, truncated=True)
    assert mat.shape == (len(x), n)
    assert mat.nnz == 2 * m * len(x)


@pytest.mark.parametrize('N', [50, 100, 200])
@pytest.mark.parametrize('sigma', [2, 3, 4])
@pytest.mark.parametrize('kernel', kernel_types)
def test_nfft_matrix_large_m(N, sigma, kernel, rseed=0):
    # truncated and non-truncated matrices should be identical at large m
    rand = np.random.RandomState(rseed)
    x = -0.5 + rand.rand(N)
    n = sigma * N
    m = n // 2
    kernel = KERNELS[kernel]

    mat1 = nfft_matrix(x, n, m, sigma, kernel, truncated=False)
    mat2 = nfft_matrix(x, n, m, sigma, kernel, truncated=True)

    assert_allclose(mat1, mat2.toarray())


@pytest.mark.parametrize('N', [50, 100, 200])
@pytest.mark.parametrize('sigma', [2, 3, 4])
def test_fourier_sum(N, sigma, rseed=0):
    # Test that the fft and direct methods match and are the right shape
    rand = np.random.RandomState(rseed)
    n = sigma * N
    ghat = rand.randn(N) + 1j * rand.randn(N)

    sum1 = fourier_sum(ghat, N, n, use_fft=False)
    sum2 = fourier_sum(ghat, N, n, use_fft=True)

    assert sum1.shape == (n,)
    assert_allclose(sum1, sum2)


@pytest.mark.parametrize('N', [50, 100, 200])
@pytest.mark.parametrize('sigma', [2, 3, 4])
def test_inv_fourier_sum(N, sigma, rseed=0):
    # Test that the fft and direct methods match and are the right shape
    rand = np.random.RandomState(rseed)
    n = sigma * N
    g = rand.randn(n) + 1j * rand.randn(n)

    sum1 = inv_fourier_sum(g, N, n, use_fft=False)
    sum2 = inv_fourier_sum(g, N, n, use_fft=True)

    assert sum1.shape == (N,)
    assert_allclose(sum1, sum2)
