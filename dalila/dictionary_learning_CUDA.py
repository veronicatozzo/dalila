from __future__ import division

import numpy as np
import logging
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils import check_random_state

from dalila.utils import _non_negative_projection, _check_non_negativity,\
                         _compute_clusters_and_silhouettes
from dalila.penalty import Penalty
from dalila.gpu_procedures import _non_negative_projection_GPU

import pycuda.gpuarray as gpuarray
from pycuda import driver
import pycuda.autoinit
import skcuda.cublas as cublas
import skcuda.misc as misc
import skcuda.linalg as linalg
linalg.init()


###############################################################################
class DictionaryLearningGPU(BaseEstimator):

    def __init__(self, k, dict_penalty=None, coeff_penalty=None,
                 dict_normalization=0, non_negativity='none',
                 random_state=None):

        self.k = k
        self.dict_penalty = dict_penalty
        self.coeff_penalty = coeff_penalty
        self.non_negativity = non_negativity
        self.dict_normalization = dict_normalization
        self.random_state = random_state
        self.X = None
        self.D = None
        self.C = None

    def fit(self, x, y=None, n_iter=20000, backtracking=0):

        # ______________ parameters control________________________________#
        x = check_array(x)
        self.X = x
        n, p = x.shape
        random_state = check_random_state(self.random_state)
        self.dict_penalty = _check_penalty(self.dict_penalty)
        self.coeff_penalty = _check_penalty(self.coeff_penalty)
        _check_number_of_atoms(self.k, p, n)
        _check_non_negativity(self.non_negativity, x)


        logging.debug("Starting procedure with " +
                      str(self.k)+" number of atoms")
        logging.debug("Initializing dictionary and coefficients..")

        # ________________optimization procedure____________________________#
        self.D, self.C = \
            self._alternating_proximal_gradient_minimization(random_state,
                n_iter=n_iter, backtracking=backtracking)

        if self.dict_normalization:
            for k in range(self.k):
                normalization_factor = np.linalg.norm(self.D[k, :])
                self.D[k, :] /= normalization_factor
                self.C[:, k] *= normalization_factor

        # __________________________final controls__________________________#
        logging.debug("Finished optimization")
        if self.non_negativity == 'both':
            assert (np.min(self.D) >= 0 and np.min(self.C) >= 0)
        if self.non_negativity == 'coeff':
            assert (np.min(self.C) >= 0)

        return self


    def objective_function_value(self, x=None, d=None, c=None):

        reconstruction = np.linalg.norm(x - c.dot(d))**2
        penalty_dictionary = self.dict_penalty.value(d)
        penalty_coefficient = self.coeff_penalty.value(c)
        return reconstruction + penalty_coefficient + penalty_dictionary





    def _alternating_proximal_gradient_minimization(self, random_state,
                                                    n_iter=20000,
                                                    backtracking=0):
        n, p = self.X.shape

        x = gpuarray.to_gpu(self.X.astype(np.float32))
        d = gpuarray.to_gpu((random_state.rand(self.k, p) * 10 - 5)
                            .astype(np.float32))
        d = _non_negative_projection_GPU(d, self.non_negativity, 'dict')
        c = gpuarray.to_gpu((random_state.rand(n, self.k) * 10 - 5)
                            .astype(np.float32))
        c = _non_negative_projection_GPU(c, self.non_negativity, 'coeff')

        norm_d = np.float32(1.1) * linalg.norm(linalg.dot(linalg.transpose(d), d))
        step_c = np.float32(1) / norm_d

        norm_c = np.float32(1.1) * linalg.norm(linalg.dot(linalg.transpose(c), c))
        step_d = np.float32(1) / norm_c

        epsilon = np.float32(1e-4)
        old_objective = linalg.norm(linalg.dot(c,d)-x)
        d_old = d.copy()
        c_old = c.copy()
        for i in range(1000):
            subtraction = misc.subtract(linalg.dot(c, d), x)
            gradient_d = linalg.dot(linalg.transpose(c), subtraction)
            gradient_c = linalg.dot(subtraction, linalg.transpose(d))

            d = self.dict_penalty. \
                apply_prox_operator(misc.subtract(d, step_d*gradient_d), step_d)
            d = _non_negative_projection_GPU(d, self.non_negativity, 'dict')
            c = self.coeff_penalty. \
                apply_prox_operator(misc.subtract(c, step_c *  gradient_c), step_c)
            c = _non_negative_projection_GPU(c, self.non_negativity, 'coeff')

            new_objective = linalg.norm(linalg.dot(c,d)-x)
            difference_objective = misc.subtract(new_objective, old_objective)
            old_objective = new_objective

            difference_d = linalg.norm(misc.subtract(d, d_old))
            difference_c = linalg.norm(misc.subtract(c,c_old))
            d_old = d.copy()
            c_old = c.copy()

            norm_d = np.float32(1.1) * linalg.norm(linalg.dot(linalg.transpose(d), d))
            step_c = np.float32(1) / norm_d

            norm_c = np.float32(1.1) * linalg.norm(linalg.dot(linalg.transpose(c), c))
            step_d = np.float32(1) / norm_c


            if (abs(difference_objective) <= epsilon and
                    difference_d <= epsilon and
                    difference_c <= epsilon):
                break

        return d.get(), c.get()



def _check_penalty(penalty):

    if penalty is None:
        return Penalty()

    if not isinstance(penalty, Penalty):
        logging.warning('The penalty is not a subclass of the '
                        'right type.')
        raise TypeError('The penalty is not a subclass of the '
                        'right type.')

    return penalty


def _check_number_of_atoms(k, p, n):
    if k is None:
        logging.warning('No number of atoms was given. '
                        'Impossible to do optimization.')
        raise ValueError('No number of atoms was given. '
                        'Impossible to do optimization.')
    if k <= 0 or k > min(n, p):
        logging.warning('The number of atoms must be greater than zero '
                        'and less than the min of the dimensions of X.')
        raise ValueError('The number of atoms must be greater than zero '
                        'and less than the min of the dimensions of X.')


def _sampling(X, random_state):  # sampling with replacement
    selected = random_state.randint(0, high=X.shape[0],
                                    size=(X.shape[0]))
    return X[selected, :]
