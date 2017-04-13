# nfft package

[![build status](
  http://img.shields.io/travis/jakevdp/nfft/master.svg?style=flat)](
 https://travis-ci.org/jakevdp/nfft/)

The ``nfft`` package is a pure-Python implementation of the non-uniform
fast Fourier transform (NFFT), based on numpy and scipy and released under
an MIT license.
For information about the NFFT algorithm, see the paper
[*Using NFFT 3 – a software library for various nonequispaced fast Fourier transforms*](http://dl.acm.org/citation.cfm?id=1555388).

The ``nfft`` package implements one-dimensional versions of the forward and
adjoint nonuniform fast Fourier transforms;

The forward transform:

![$f_j = \sum_{j=0}^{M-1} f_k e^{-2\pi i k x_j}$](figures/forward-formula.png)

And the adjoint transform:

![$\hat{f}_k = \sum_{j=0}^{M-1} f_j e^{2\pi i k x_j}$](figures/adjoint-formula.png)

In both cases, the wavenumbers *k* are on a regular grid from -N/2 to N/2,
while the data values *x_j* are irregularly spaced between -1/2 and 1/2.

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