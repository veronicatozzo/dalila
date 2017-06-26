from nose.tools import *
import numpy as np

from dalila.representation_learning import RepresentationLearning
from dalila.dataset_generator import synthetic_data_non_negative


@raises(TypeError)
def wrong_penalty_type_test():
    X, _, D = synthetic_data_non_negative()

    estimator = RepresentationLearning(coeff_penalty="")
    estimator.fit(X, D)


def non_negative_parameter_test():
    X, _, D = synthetic_data_non_negative()

    estimator = RepresentationLearning(non_negativity=True)
    estimator.fit(X, D)
    C = estimator.coefficients()

    assert (np.min(C) >= 0)