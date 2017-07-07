DALILA (DictionAry LearnIng LibrAry)
====================================

**DALILA** is a Dictionary Learning Library whose purpose is to find a decomposition of an input matrix **X** into two other matrices **D** and **C** which are respectively the *dictionary* which contains the basic atoms and the *coefficients* that are weights for the atoms. The linear combination of atoms weighted with the coefficients give an approximation of the original signal.

We propose a generic optimization algorithm that can optimize the functional
with different penalties both on the dictionary and on the coefficients.
The algorithm works for unsupervised dictionary learning and sparse coding.

The library allows to run some of its computationally expensive parts in parallel
on the same machine or distributing the tasks with dask (http://dask.pydata.org/en/latest/index.html).

Unsupervised dictionary learning
---------------------------------
Optimization of a dictionary learning problem in the following form

            ||X - DC|| + c(C) + d(D)

    
the penalties on **C** and **D** can be of different types an precisely we offer the possibility to use:

- L1-norm    
- L2-norm
- elastic-net
- Group lasso
- L0 norm
- LInf norm
- Total Variation

besides we offere the possibility to impose non-negativy contraints on the decomposition
on both the matrices, on only the coefficients or for none of them.

Sparse coding
--------------
Optimization of a problem in the following form

            ||X - DC|| + c(C)

where the dictionary is known and fixed. The penalties that can be used on the coefficients
are the same listed above.

Stability dictionary learning
-----------------------------
We offer a class, called StabilityDictionaryLearning that executes the dictionary
learning algorithm and iteratively clusters the atoms to find a stable solution
w.r.t. the noise in the data.

Cross-validation
----------------
The library contains a procedure to analyse which is the best number of atoms
to decompose the signal matrix and which are the best regularization parameters.
The score for cross-validation is the BIC (Bayesian Information Criterion) value
computed on the objective function value.


## Installation

**dalila** supports Python 2.7 and Python3.6

#### Pip installation
`$ pip install dalila`

#### Conda installation
`$ conda install dalila`

#### Installing from sources
```bash
$ git clone https://github.com/slipguru/dalila
$ cd dalila
$ python setup.py install
```


## Quick start

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

