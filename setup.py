from __future__ import print_function
import sys
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]


try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import scipy
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)

try:
    import sklearn
except ImportError:
    print('sklearn is required during installation')
    sys.exit(1)

try:
    import matplotlib
except ImportError:
    print('matplotlib is required during installation')
    sys.exit(1)


try:
    import bintrees
except ImportError:
    print('bintrees is required during installation')
    sys.exit(1)

try:
    import itertools
except ImportError:
    print('itertools is required during installation')
    sys.exit(1)

try:
    import logging
except ImportError:
    print('logging is required during installation')
    sys.exit(1)

try:
    import multiprocessing
except ImportError:
    print('multiprocessing is required during installation')
    sys.exit(1)

try:
    import distributed
except ImportError:
    print('distributed is required during installation')
    sys.exit(1)


setup(name='dalila',
      version='1.0.0',
      description='A library for dictionary learning decomposition',
      long_description='A library for the decomposition of a matrix of signals.'
                       'It includes sparse coding algorithm and parameter '
                       'research procedures.',
      url='https://github.com/slipguru/dalila',
      author='Veronica Tozzo',
      author_email='veronica.tozzo@dibris.unige.it',
      license='FreeBSD',
      classifiers={
          'Development Status :: 0',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'Programming Language :: Python',
          'License :: OSI Approved :: BSD License',
          'Topic :: Software Development',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Operating System :: POSIX',
          'Operating System :: Unix'},
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      )
