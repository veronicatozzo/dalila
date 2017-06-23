import numpy as np

from nose.tools import *

from dalila.dictionary_learning import DictionaryLearning,\
                                       StabilityDictionaryLearning
from dalila.dataset_generator import *


@raises(TypeError)
def wrong_penalty_type_coeff_test():
    X = np.random.rand(100,100)

    estimator = DictionaryLearning(k=5, coeff_penalty="")
    estimator.fit(X)


@raises(TypeError)
def wrong_penalty_type_dict_test():
    X = np.random.rand(100,100)

    estimator = DictionaryLearning(k=5, dict_penalty="")
    estimator.fit(X)


@raises(ValueError)
def none_atoms_test():
    X = np.random.rand(100,100)

    estimator = DictionaryLearning(k=None, dict_penalty="")
    estimator.fit(X)


@raises(ValueError)
def negative_atoms_test():
    X = np.random.rand(100,100)

    estimator = DictionaryLearning(k=-1, dict_penalty="")
    estimator.fit(X)


@raises(ValueError)
def non_negative_parameter_test():
    X = np.random.rand(100, 100)

    estimator = DictionaryLearning(k=5, non_negativity="not_considered")
    estimator.fit(X)


@raises(ValueError)
def non_negative_parameter_but_negative_matrix_test():
    X = np.random.rand(100, 100) - 50

    estimator = DictionaryLearning(k=5, non_negativity="both")
    estimator.fit(X)


def non_negativity_decomposition_on_both_test():
    X = np.abs(np.random.rand(100, 100))

    estimator = DictionaryLearning(k=5, non_negativity="both")
    estimator.fit(X)

    C, D = estimator.decomposition()

    assert np.min(C) >= 0
    assert np.min(D) >= 0


def non_negativity_decomposition_on_coefficients_test():
    X = np.random.rand(100, 100)-100

    estimator = DictionaryLearning(k=5, non_negativity="coeff")
    estimator.fit(X)

    C, D = estimator.decomposition()

    assert np.min(C) >= 0
    assert np.min(D) < 0


def right_number_of_atoms_test():
    X = np.random.rand(100, 100)

    estimator = DictionaryLearning(k=10)
    estimator.fit(X)

    C, D = estimator.decomposition()

    assert C.shape[1] == 10
    assert D.shape[0] == 10


def normalization_test():
    X = np.random.rand(100, 100)-100

    estimator = DictionaryLearning(k=10, dict_normalization=1)
    estimator.fit(X)

    C, D = estimator.decomposition()

    for i in range(D.shape[0]):
        np.testing.assert_approx_equal(np.linalg.norm(D[0, :]), 1)


# STABILITY DL TESTS
@raises(TypeError)
def stability_wrong_penalty_type_coeff_test():
    X = np.random.rand(100,100)

    estimator = StabilityDictionaryLearning(k=5, coeff_penalty="")
    estimator.fit(X)


@raises(TypeError)
def stability_wrong_penalty_type_dict_test():
    X = np.random.rand(100,100)

    estimator = StabilityDictionaryLearning(k=5, dict_penalty="")
    estimator.fit(X)


@raises(ValueError)
def stability_none_atoms_test():
    X = np.random.rand(100,100)

    estimator = StabilityDictionaryLearning(k=None, dict_penalty="")
    estimator.fit(X)


@raises(ValueError)
def stability_negative_atoms_test():
    X = np.random.rand(100,100)

    estimator = StabilityDictionaryLearning(k=-1, dict_penalty="")
    estimator.fit(X)


@raises(ValueError)
def stability_non_negative_parameter_test():
    X = np.random.rand(100, 100)

    estimator = StabilityDictionaryLearning(k=5,
                                            non_negativity="not_considered")
    estimator.fit(X)


@raises(ValueError)
def stability_non_negative_parameter_but_negative_matrix_test():
    X, _, _ = synthetic_data_negative()

    estimator = StabilityDictionaryLearning(k=5, non_negativity="both")
    estimator.fit(X)


# def S_non_negativity_decomposition_on_both_test():
#     X, _, _ = synthetic_data_non_negative()
#
#     estimator = StabilityDictionaryLearning(k=5, non_negativity="both")
#     estimator.fit(X)
#
#     C, D = estimator.decomposition()
#
#     assert np.min(C) >= 0
#     assert np.min(D) >= 0
#
#
# def S_non_negativity_decomposition_on_coefficients_test():
#     X, _, _= synthetic_data_negative()
#
#     estimator = StabilityDictionaryLearning(k=5, non_negativity="coeff")
#     estimator.fit(X)
#
#     C, D = estimator.decomposition()
#
#     assert np.min(C) >= 0
#     assert np.min(D) < 0
#
#
# def S_right_number_of_atoms_test():
#     X, _, _ = synthetic_data_negative()
#
#     estimator = StabilityDictionaryLearning(k=10)
#     estimator.fit(X)
#
#     C, D = estimator.decomposition()
#
#     assert C.shape[1] == 10
#     assert D.shape[0] == 10
#
#
# def S_normalization_test():
#     X,_,_ = synthetic_data_negative()
#
#     estimator = StabilityDictionaryLearning(k=10, dict_normalization=1)
#     estimator.fit(X)
#
#     C, D = estimator.decomposition()
#
#     for i in range(D.shape[0]):
#         np.testing.assert_approx_equal(np.linalg.norm(D[0, :]), 1)