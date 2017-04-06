import numpy as np

import pytest
from numpy.testing import assert_allclose, assert_array_less

from ..core import indft, infft


def generate_data(N, amp=1, rseed=0):
    rand = np.random.RandomState(rseed)
    x = -0.5 + rand.rand(N)
    f = amp * (rand.randn(N) + 1j * rand.rand(N))
    return x, f


@pytest.mark.parametrize('N', [50, 100, 200])
@pytest.mark.parametrize('Nf', [50, 100, 200])
@pytest.mark.parametrize('sigma', [2, 3, 4, 5])
@pytest.mark.parametrize('use_fft', [True, False])
@pytest.mark.parametrize('use_sparse', [True, False])
def test_infft_slow(N, Nf, sigma, use_fft, use_sparse):
    x, f = generate_data(N)

    direct = indft(x, f, Nf)
    approx = infft(x, f, Nf, sigma=sigma, use_fft=use_fft, use_sparse=use_sparse)

    assert_allclose(direct, approx)


@pytest.mark.parametrize('N', [50, 100, 200])
@pytest.mark.parametrize('Nf', [50, 100, 200])
@pytest.mark.parametrize('tol', [1E-4, 1E-8, 1E-12])
@pytest.mark.parametrize('amp', [1, 10, 100])
@pytest.mark.parametrize('sigma', [2, 3, 4, 5])
def test_tol(N, Nf, tol, amp, sigma):
    x, f = generate_data(N, amp=amp)
    direct = indft(x, f, Nf)
    approx = infft(x, f, Nf, sigma=sigma, tol=tol)
    observed_diff = abs(direct - approx)
    assert observed_diff.max() < tol * abs(f).sum()
