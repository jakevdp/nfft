from __future__ import division

import numpy as np
from numpy.lib import scimath
from scipy import signal, special


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

    def C(self, m, sigma):
        return 4 * np.exp(-m * np.pi * (1 - 1. / (2 * sigma - 1)))

    def m_from_C(self, C, sigma):
        return np.ceil(-np.log(0.25 * C) / (np.pi * (1 - 1 / (2 * sigma - 1))))


# Kaiser-Bessel Kernel. Seems to have some issues with overflow

# class KaiserBesselKernel(NFFTKernel):
#     def _b(self, sigma, m):
#         return np.pi * (2 - 1 / sigma)
#
#     def phi(self, x, n, m, sigma):
#         b = self._b(sigma, m) / np.pi
#         arg = scimath.sqrt(n ** 2 * x ** 2 - m ** 2)
#         return b * np.sinc(b * arg).real
#
#     def phi_hat(self, k, n, m, sigma):
#         b = self._b(sigma, m)
#         result = (1 / n) * special.i0(m * np.sqrt(b ** 2 -
#                                                   (2 * np.pi * k / n) ** 2))
#         result[abs(k) > n * (1 - 0.5 / sigma)] = 0
#         return result
#
#     def C(self, m, sigma):
#         raise NotImplementedError()
#
#     def m_from_C(self, C, sigma):
#         raise NotImplementedError()

# B Spline Kernel. Seems to have issues with stability at larger m

# class BSplineKernel(NFFTKernel):
#     def phi(self, x, n, m, sigma):
#         return signal.bspline(n * x, 2 * m)
#
#     def phi_hat(self, k, n, m, sigma):
#         return (1 / n) * np.sinc(k / n) ** (2 * m)
#
#     def C(self, m, sigma):
#         return 4 * (1. / (2 * sigma - 1)) ** (2 * m)
#
#     def m_from_C(self, C, sigma):
#         return 0.5 * np.log(0.25 * C) / np.log(1. / (2 * sigma - 1))


KERNELS = dict(gaussian=GaussianKernel())
