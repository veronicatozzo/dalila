from nose.tools import *
import numpy as np

from dalila.dl.representation_learning import RepresentationLearning
from dalila.dl.dataset_generator import synthetic_data_non_negative


@raises(TypeError)
def wrong_penalty_type_test():
    X, _, D = synthetic_data_non_negative()

    estimator = RepresentationLearning(D, coeff_penalty="")
    estimator.fit(X)


def non_negative_parameter_test():
    X, _, D = synthetic_data_non_negative()

    estimator = RepresentationLearning(D, non_negativity=True)
    estimator.fit(X)
    C = estimator.coefficients()

    assert (np.min(C) >= 0)
