# nfft package

[![build status](
  http://img.shields.io/travis/jakevdp/nfft/master.svg?style=flat)](
 https://travis-ci.org/jakevdp/nfft/)

The ``nfft`` package is a pure-Python implementation of the non-equispaced
fast Fourier transform (NFFT), based on numpy and scipy and released under
an MIT license.
For information about the NFFT algorithm, see the paper
[*Using NFFT 3 – a software library for various nonequispaced fast Fourier transforms*](http://dl.acm.org/citation.cfm?id=1555388).



## About

The ``nfft`` package implements one-dimensional versions of the forward and
adjoint non-equispaced fast Fourier transforms;

The forward transform:

![$f_j = \sum_{j=0}^{M-1} f_k e^{-2\pi i k x_j}$](figures/forward-formula.png)

And the adjoint transform:

![$\hat{f}_k = \sum_{j=0}^{M-1} f_j e^{2\pi i k x_j}$](figures/adjoint-formula.png)

In both cases, the wavenumbers *k* are on a regular grid from -N/2 to N/2,
while the data values *x_j* are irregularly spaced between -1/2 and 1/2.

The direct and fast version of these algorithms are implemented in the following
functions:

- ``nfft.ndft``: direct forward non-equispaced Fourier transform
- ``nfft.nfft``: fast forward non-equispaced Fourier transform
- ``nfft.ndft_adjoint``: direct adjoint non-equispaced Fourier transform
- ``nfft.nfft_adjoint``: fast adjoint non-equispacedFourier transform



## Comparison to pynfft

Another option for computing the NFFT in Python is to use the
[pynfft](https://github.com/ghisvail/pyNFFT/) package, which provides a
Python wrapper to the C library referenced in the above paper.

The advantage of ``pynfft`` is that it provides a more complete set of
routines, including multi-dimensional NFFTs and various computing strategies.

The disadvantage is that ``pynfft`` is GPL-licensed (and thus can't be used
in much of the more permissively licensed Python scientific world), and has
a much more complicated set of dependencies.

Performance-wise, ``nfft`` and ``pynfft`` are comparable, with this pure-Python
package being up to a factor of 2 faster in most cases of interest
(see [Benchmarks.ipynb](notebooks/Benchmarks.ipynb) for some simple
benchmarks).

If you're curious how this is implemented, see the [Implementation Walkthrough](notebooks/ImplementationWalkthrough.ipynb) notebook.



## Installation

The ``nfft`` package can be installded directly from the Python Package Index:

```
$ pip install nfft
```

Dependencies are [numpy](http://www.numpy.org), [scipy](http://www.scipy.org), and [pytest](http://www.pytest.org).




## Testing

Unit tests can be run using [pytest](http://pytest.org):

```
$ pytest --pyargs nfft
```



## Examples

For some basic usage examples, see the notebooks in the [notebooks](notebooks)
directory.




## License

This code is released under the [MIT License](LICENSE). For more information,
see the [Open Source Initiative](https://opensource.org/licenses/MIT)