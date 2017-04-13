# Version info: don't use any relative imports here, because setup.py
# runs this as a standalone script to extract the following information

from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 2
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "nfft: lightweight non-equispaced fast Fourier transform"
# Long description will go up on the pypi page
long_description = """

nfft package
============
nfft is a lightweight implementation of the non-equispaced fast Fourier
transform. Its performance is comparable to that of pynfft_, but it contains
no compiled code and requires no links to external C libraries, beyond standard
dependencies on numpy_ and scipy_.

For more information and links to usage examples, please see the
repository README_.

.. _README: https://github.com/jakevdp/nfft/blob/master/README.md

.. _pynfft: https://pypi.python.org/pypi/pyNFFT

.. _numpy: https://www.numpy.org

.. _scipy: https://www.scipy.org

License
=======
``nfft`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.
"""

NAME = "nfft"
MAINTAINER = "Jake VanderPlas"
MAINTAINER_EMAIL = "jakevdp@uw.edu"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/jakevdp/nfft/"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Jake VanderPlas"
AUTHOR_EMAIL = "jakevdp@uw.edu"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {}
REQUIRES = ["numpy", "scipy", "pytest"]
