import numpy as np

from numpy.testing import assert_allclose
import pytest

from ..kernels import KERNELS
kernel_types = sorted(KERNELS.keys())


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
    
