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

    def estimate_m(self, tol, N, sigma):
        # TODO: this should be computed in terms of the L1-norm of the true
        #   Fourier coefficients... see p. 11 of
        #   https://www-user.tu-chemnitz.de/~potts/nfft/guide/nfft3.pdf
        #   Need to think about how to estimate the value of m more accurately
        C = tol / N
        return self.m_from_C(C, sigma)


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


# TODO: implement some other kernels from the literature


KERNELS = dict(gaussian=GaussianKernel())
