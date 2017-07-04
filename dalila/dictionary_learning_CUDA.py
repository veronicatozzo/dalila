from __future__ import division

import numpy as np
import logging
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils import check_random_state

from dalila.utils import non_negative_projection, _check_non_negativity,\
                         _compute_clusters_and_silhouettes
from dalila.penalty import Penalty


###############################################################################
class DictionaryLearning(BaseEstimator):

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
        """

        Parameters
        ----------
        x : array-like, shape=(n_samples, n_features)
            The matrix to be decomposed.

        d: array_like, shape=(n_atoms, n_features)
            The dictionary.

        c: array-like, shape=(n_samples, n_atoms)
            The matrix of coefficients

        If one of the three is None the internal decomposition is taken, if no
        decomposition is available NaN is returned.

        Returns
        -------
        float:
            The value of the objective function.
        """
        if x is None:
            x = self.X
            if x is None:
                logging.warning('Called objective function value with no'
                                'matrices before calling fit. \n'
                                'Impossible to return an objective function '
                                'value')
                return float('NaN')

        if d is None:
            d = self.D
        if c is None:
            c = self.C

        reconstruction = np.linalg.norm(x - c.dot(d))**2
        penalty_dictionary = self.dict_penalty.value(d)
        penalty_coefficient = self.coeff_penalty.value(c)
        return reconstruction + penalty_coefficient + penalty_dictionary





    def _alternating_proximal_gradient_minimization(self, random_state,
                                                    n_iter=20000,
                                                    backtracking=0):
        x = self.X
        n, p = x.shape
        d = non_negative_projection(random_state.rand(self.k, p)*10-5,
                                    self.non_negativity, 'dict')
        c = non_negative_projection(random_state.rand(n, self.k)*10-5,
                                    self.non_negativity, 'coeff')

        gamma_c = 1.1
        gamma_d = gamma_c
        step_d, step_c = _step_lipschitz(d, c,
                                         gamma_d=gamma_d, gamma_c=gamma_c)
        epsilon = 1e-4
        objective = self.objective_function_value(d=d, c=c)

        d_old = d
        c_old = c
        logging.debug("Starting optimization")
        for i in range(n_iter):
            gradient_d = c.T.dot(c.dot(d) - x)
            gradient_c = (c.dot(d) - x).dot(d.T)

            if backtracking:
                d, c = self._update_with_backtracking(d, c, gradient_d,
                                                      gradient_c, step_d,
                                                      step_c)
            else:
                d, c = self._simple_update(d, c, gradient_d, gradient_c,
                                           step_d, step_c)

            new_objective = self.objective_function_value(d=d, c=c)
            difference_objective = new_objective - objective
            objective = new_objective
            difference_d = np.linalg.norm(d - d_old)
            difference_c = np.linalg.norm(c - c_old)
            d_old = d
            c_old = c

            step_d, step_c = _step_lipschitz(d, c,
                                             gamma_c=gamma_c, gamma_d=gamma_d)

            logging.debug("Iteration: "+str(i))
            logging.debug("Objective function: "+str(objective))
            logging.debug("Difference between new objective and "
                          "previous one: "+str(difference_objective))
            logging.debug("Difference between previous and new dictionary: " +
                          str(difference_d))
            logging.debug("Difference between previous and new coefficients:" +
                          str(difference_c)+"`\n\n")

            assert ((not np.isnan(difference_objective)) and
                    (not np.isinf(difference_objective)) and
                    (abs(difference_objective) < 1e+20))

            if (abs(difference_objective) <= epsilon and
                    difference_d <= epsilon and
                    difference_c <= epsilon):
                break

        return d, c

    def _simple_update(self, d, c, gradient_d, gradient_c,
                       step_d, step_c):
        d = self.dict_penalty. \
            apply_prox_operator(d - step_d * gradient_d, gamma=step_d)
        d = non_negative_projection(d, self.non_negativity, 'dict')

        c = self.coeff_penalty. \
            apply_prox_operator(c - step_c * gradient_c, gamma=step_c)
        c = non_negative_projection(c, self.non_negativity, 'coeff')
        return d, c







