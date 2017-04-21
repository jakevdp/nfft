import numpy as np

from numpy.testing import assert_allclose
import pytest

from ..kernels import KERNELS, NDKERNELS
kernel_types = sorted(KERNELS.keys())


@pytest.mark.parametrize('method', ['phi', 'phi_hat'])
@pytest.mark.parametrize('kernel', kernel_types)
def test_ndkernel_input_dimensions(kernel, method, n=100, m=10, sigma=2):
    kernel = NDKERNELS.get(kernel)
    method = getattr(kernel, method)

    # test 1D variants
    x = np.random.rand(12)
    k1 = method(x, n, m, sigma)
    k2 = method(x[:, None], n, m, sigma)
    assert_allclose(k1, k2)

    k3 = method(x.reshape(3, 4, 1), n, m, sigma)
    assert_allclose(k2, k3.ravel())

    # test 2D variants
    x = np.random.rand(12, 3)
    k1 = method(x, n, m, sigma)
    k2 = method(x.reshape(3, 4, 3), n, m, sigma)
    assert_allclose(k1, k2.ravel())

    k3 = method(x, n + np.zeros(3), m + np.zeros(3), sigma)
    assert_allclose(k1, k3)


@pytest.mark.parametrize('kernel', kernel_types)
@pytest.mark.parametrize('sigma', [2, 3])
@pytest.mark.parametrize('n', [1000, 2000])
@pytest.mark.parametrize('m', [50, 100])
def test_kernel_fft(kernel, sigma, n, m):
    kernel = KERNELS[kernel]

    x = np.linspace(-0.5, 0.5, n, endpoint=False)
    f = kernel.phi(x, n, m, sigma)

    k = -(n // 2) + np.arange(n)
    f_hat = kernel.phi_hat(k, n, m, sigma)

    f_fft = (1. / n) * np.fft.fftshift(np.fft.fft(np.fft.fftshift(f)))

    assert_allclose(f_hat, f_fft, atol=1E-12)


@pytest.mark.parametrize('kernel', kernel_types)
@pytest.mark.parametrize('sigma', [2, 3])
def test_kernel_m_C(kernel, sigma):
    kernel = KERNELS[kernel]
    m = np.arange(1, 100)
    C = kernel.C(m, sigma)
    m2 = kernel.m_from_C(C, sigma).astype(int)
    assert_allclose(m, m2, atol=1)  # atol=1 for float->int rounding errors
