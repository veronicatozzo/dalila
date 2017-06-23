import numpy as np
from nose.tools import *

from dalila.parameters_research import tune_parameters_DL
from dalila.dictionary_learning import DictionaryLearning
from dalila.penalty import L1Penalty
from dalila.dataset_generator import synthetic_data_negative


@raises(TypeError)
def wrong_estimator_test():
    X = np.random.rand(10, 10)
    res = tune_parameters_DL(X, "")


@raises(ValueError)
def wrong_range1_test():
    X, _, _ = synthetic_data_negative()
    estimator = DictionaryLearning(k=5, coeff_penalty=L1Penalty(1),
                                   dict_penalty=(L1Penalty(2)),
                                   non_negativity="coeff")
    res = tune_parameters_DL(X, estimator, analysis=1, distributed=0,
                             dict_penalty_range=(0.0001, 1),
                             coeff_penalty_range=(0.0001, 1))


@raises(ValueError)
def wrong_range2_test():
    X, _, _= synthetic_data_negative()
    estimator = DictionaryLearning(k=5, coeff_penalty=L1Penalty(1),
                                   dict_penalty=(L1Penalty(2)),
                                   non_negativity="coeff")
    res = tune_parameters_DL(X, estimator, analysis=1, distributed=0,
                             dict_penalty_range=(-1, 1, 5),
                             coeff_penalty_range=(-1, 1, 5))

