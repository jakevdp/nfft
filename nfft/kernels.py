from __future__ import division

import numpy as np


class NFFTKernel(object):
    def phi(self, x, n, m, sigma):
        raise NotImplementedError()

    def phi_hat(self, x, n, m, sigma):
        raise NotImplementedError()

    def C(self, m, sigma):
        raise NotImplementedError()

    def m_from_C(self, C, sigma):
        raise NotImplementedError()


class GaussianKernel(NFFTKernel):
    def _b(self, sigma, m):
        return (2 * sigma * m) / ((2 * sigma - 1) * np.pi)

    def phi(self, x, n, m, sigma):
        b = self._b(sigma, m)
        return np.exp(-(n * x) ** 2 / b) / np.sqrt(np.pi * b)

    def phi_hat(self, k, n, m, sigma):
        b = self._b(sigma, m)
        return np.exp(-b * (np.pi * k / n) ** 2) / n

    def C(self, sigma, m):
        return 4 * np.exp(-m * np.pi * (1 - 1. / (2 * sigma - 1)))

    def m_from_C(self, C, sigma):
        return np.ceil(-np.log(0.25 * C) / (np.pi * (1 - 1 / (2 * sigma - 1))))


KERNELS = dict(gaussian=GaussianKernel())
