import numpy as np
from nose.tools import *

from dalila.parameters_research import tune_parameters_DL, tune_parameters_RL
from dalila.dictionary_learning import DictionaryLearning
from dalila.representation_learning import RepresentationLearning
from dalila.penalty import L1Penalty
from dalila.dataset_generator import synthetic_data_negative


@raises(TypeError)
def wrong_estimator_test():
    X = np.random.rand(10, 10)
    tune_parameters_DL(X, "")


@raises(ValueError)
def wrong_range1_test():
    X, _, _ = synthetic_data_negative()
    estimator = DictionaryLearning(k=5, coeff_penalty=L1Penalty(1),
                                   dict_penalty=(L1Penalty(2)),
                                   non_negativity="coeff")
    tune_parameters_DL(X, estimator, analysis=1, distributed=0,
                             dict_penalty_range=(0.0001, 1),
                             coeff_penalty_range=(0.0001, 1))


@raises(ValueError)
def wrong_range2_test():
    X, _, _= synthetic_data_negative()
    estimator = DictionaryLearning(k=5, coeff_penalty=L1Penalty(1),
                                   dict_penalty=(L1Penalty(2)),
                                   non_negativity="coeff")
    tune_parameters_DL(X, estimator, analysis=1, distributed=0,
                             dict_penalty_range=(-1, 1, 5),
                             coeff_penalty_range=(-1, 1, 5))


@raises(TypeError)
def wrong_estimator_RL_test():
    X = np.random.rand(10, 10)
    tune_parameters_RL(X, "")


@raises(ValueError)
def none_estimator_test():
    X = np.random.rand(10, 10)
    tune_parameters_RL(X, None)


@raises(ValueError)
def wrong_range_test():
    X, _, D = synthetic_data_negative()
    estimator = RepresentationLearning(D, penalty=L1Penalty(1),
                                       non_negativity=1)
    tune_parameters_RL(X, estimator, distributed=0,
                             coeff_penalty_range=(0.0001, 1))




