import numpy as np

from dalila.dataset_generator import *


def group_lasso_generator_test():
    S, C, D = group_lasso_dataset_generator()
    groups = [[0, 1, 3], [2, 4, 5]]

    assert S.shape == (100, 100)
    assert D.shape == (6, 100)
    assert C.shape == (100, 6)

    assert np.sum(C[0:50, groups[1]]) == 0
    assert np.sum(C[50:100, groups[0]]) == 0


def synthetic_data_negative_generator_test():
    S, C, D = synthetic_data_negative(n_samples=50, n_features=10000)
    assert S.shape == (50, 10000)
    assert D.shape == (5, 10000)
    assert C.shape == (50, 5)


def synthetic_data_non_negative_generator_test():
    S, C, D = synthetic_data_non_negative()
    assert S.shape == (80, 96)
    assert D.shape == (7, 96)
    assert C.shape == (80, 7)


