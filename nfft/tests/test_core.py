import numpy as np

import pytest
from numpy.testing import assert_allclose

from .. import ndft, nfft, ndft_adjoint, nfft_adjoint
from ..kernels import KERNELS
kernel_types = sorted(KERNELS.keys())


def generate_data(N, Nf, amp=1, rseed=0):
    rand = np.random.RandomState(rseed)
    x = -0.5 + rand.rand(N)
    f_hat = amp * (rand.randn(Nf) + 1j * rand.rand(Nf))
    return x, f_hat


def generate_adjoint_data(N, amp=1, rseed=0):
    rand = np.random.RandomState(rseed)
    x = -0.5 + rand.rand(N)
    f = amp * (rand.randn(N) + 1j * rand.rand(N))
    return x, f


@pytest.mark.parametrize('N', [50, 100, 200])
@pytest.mark.parametrize('Nf', [50, 100, 200])
@pytest.mark.parametrize('sigma', [2, 3, 4, 5])
@pytest.mark.parametrize('kernel', kernel_types)
def test_nfft_slow(N, Nf, sigma, kernel):
    x, f_hat = generate_data(N, Nf)

    direct = ndft(x, f_hat)
    approx = nfft(x, f_hat, sigma=sigma, tol=1E-8, kernel=kernel)

    assert_allclose(direct, approx)


@pytest.mark.parametrize('N', [50, 100, 200])
@pytest.mark.parametrize('Nf', [50, 100, 200])
@pytest.mark.parametrize('tol', [1E-4, 1E-8, 1E-12])
@pytest.mark.parametrize('amp', [1, 10, 100])
@pytest.mark.parametrize('sigma', [2, 3, 4, 5])
@pytest.mark.parametrize('kernel', kernel_types)
def test_nfft_tol(N, Nf, tol, amp, sigma, kernel):
    x, f_hat = generate_data(N, Nf, amp=amp)

    direct = ndft(x, f_hat)
    approx = nfft(x, f_hat, sigma=sigma, tol=tol, kernel=kernel)

    observed_diff = abs(direct - approx)
    assert observed_diff.max() < tol * abs(direct).sum()


@pytest.mark.parametrize('N', [50, 100, 200])
@pytest.mark.parametrize('Nf', [50, 100, 200])
@pytest.mark.parametrize('sigma', [2, 3, 4, 5])
@pytest.mark.parametrize('kernel', kernel_types)
def test_nfft_adjoint_slow(N, Nf, sigma, kernel):
    x, f = generate_adjoint_data(N)

    direct = ndft_adjoint(x, f, Nf)
    approx = nfft_adjoint(x, f, Nf, sigma=sigma, tol=1E-8, kernel=kernel)

    assert_allclose(direct, approx)


@pytest.mark.parametrize('N', [50, 100, 200])
@pytest.mark.parametrize('Nf', [50, 100, 200])
@pytest.mark.parametrize('tol', [1E-4, 1E-8, 1E-12])
@pytest.mark.parametrize('amp', [1, 10, 100])
@pytest.mark.parametrize('sigma', [2, 3, 4, 5])
@pytest.mark.parametrize('kernel', kernel_types)
def test_nfft_adjoint_tol(N, Nf, tol, amp, sigma, kernel):
    x, f = generate_adjoint_data(N, amp=amp)
    direct = ndft_adjoint(x, f, Nf)
    approx = nfft_adjoint(x, f, Nf, sigma=sigma, tol=tol, kernel=kernel)
    observed_diff = abs(direct - approx)
    assert observed_diff.max() < tol * abs(direct).sum()
