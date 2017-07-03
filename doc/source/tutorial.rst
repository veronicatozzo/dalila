.. _tutorial:

Quick start tutorial
====================
DALILA may be installed using standard Python tools (with
administrative or sudo permissions on GNU-Linux platforms)::

    $ pip install dalila

    or

    $ conda install dalila

Installation from sources
-------------------------
If you like to manually install dalila  you can clone our
`GitHub repository <https://github.com/slipguru/dalila>`_::

   $ git clone https://github.com/slipguru/dalila.git

From here, you can follow the standard Python installation step::

    $ python setup.py install





Examples
-------

### 1. Dictionary learning
```python
from dalila.dictionary_learning import DictionaryLearning
from dalila.penalty import L1Penalty, L2Penalty
from dalila.dataset_generator import synthetic_data_non_negative

X, _, _= synthetic_data_non_negative()
n_atoms = 7
coeff_penalty = L1Penalty(1.) # 1. is the regularization parameter
dict_penalty = L2Penalty(0.1) # 0.1 is the regularization parameter
estimator = DictionaryLearning(k=n_atoms, coeff_penalty=coeff_penalty,
                               dict_penalty=dict_penalty,
                               non_negativity="none")
estimator.fit(X)
C, D = estimator.decomposition()
```

### 2. Sparse coding
```python
from dalila.representation_learning import RepresentationLearning
from dalila.penalty import L1Penalty
from dalila.dataset_generator import synthetic_data_non_negative

X, _, D = synthetic_data_non_negative()
penalty = L1Penalty(1.) # 1. is the regularization parameter
estimator = RepresentationLearning(D, penalty=penalty,
                                   non_negativity=True)
estimator.fit(X)
C = estimator.coefficients()
```

### 3. Parameters research and cross-validation
```python
from dalila.dictionary_learning import DictionaryLearning
from dalila.penalty import L1Penalty
from dalila.parameters_research import tune_parameters_DL
from dalila.dataset_generator import synthetic_data_negative

X, _, _ = synthetic_data_negative()
estimator = DictionaryLearning(k=5, coeff_penalty=L1Penalty(1),
                               dict_penalty=(L1Penalty(2)),
                               non_negativity="coeff")
res = tune_parameters_DL(X, estimator, analysis=1, distributed=0)

```