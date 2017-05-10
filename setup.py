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


#add other required packages if needed


setup(name='dalila',
      version='1.0.0',
      description='A library for dictionary learning decomposition',
      long_description='A library for the decomposition of a matrix of signals '
                       'in an unsupervised and supervised scenario. '
                       'It includes a pipeline for the analysis of the best number'
                       'of atoms to use.',
      url='https://github.com/veronicatozzo/DaLiLa',
      author='Veronica Tozzo',
      author_email='veronica.tozzo@dibris.unige.it',
      #license='boh',
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
      #scripts=''
      )
