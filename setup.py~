from __future__ import print_function
import sys
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

setup(name='dalila',
      version='1.0.1',
      description='A library for dictionary learning decomposition',
      long_description='A library for the decomposition of a matrix of signals.'
                       'It includes sparse coding algorithm and parameter '
                       'research procedures.',
      url='https://github.com/slipguru/dalila',
      author='Veronica Tozzo',
      author_email='veronica.tozzo@dibris.unige.it',
      license='FreeBSD',
      classifiers={
          'Development Status :: 4 - Beta',
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
