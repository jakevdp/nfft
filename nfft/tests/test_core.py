import numpy as np

import pytest
from numpy.testing import assert_allclose

from ..core import indft, infft


def generate_data(N, amp=1, rseed=0):
    rand = np.random.RandomState(rseed)
    x = -0.5 + rand.rand(N)
    f = amp * (rand.randn(N) + 1j * rand.rand(N))
    return x, f


@pytest.mark.parametrize('N', [50, 100, 200])
@pytest.mark.parametrize('sigma', [2, 3, 4, 5])
@pytest.mark.parametrize('use_fft', [True, False])
@pytest.mark.parametrize('use_sparse', [True, False])
def test_infft_slow(N, sigma, use_fft, use_sparse):
    x, f = generate_data(N)

    direct = indft(x, f, N)
    slow = infft(x, f, N, sigma=sigma, use_fft=use_fft, use_sparse=use_sparse)

    assert_allclose(direct, slow)
