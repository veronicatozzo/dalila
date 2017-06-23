from dalila.dictionary_learning import DictionaryLearning
from dalila.penalty import L1Penalty
import numpy as np


def test():
    X = np.random.rand(100,100)

    estimator = DictionaryLearning(k=5, coeff_penalty=L1Penalty(0.1))
    estimator.fit(X)

    C, D = estimator.decomposition()

    assert(True)