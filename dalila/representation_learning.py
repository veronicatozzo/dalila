from __future__ import print_function, division

import numpy as np
import logging

from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils import check_random_state

from dalila.utils import non_negative_projection
from dalila.dictionary_learning import _check_penalty


class RepresentationLearning(BaseEstimator):
    """ An estimator for finding coefficients given the dictionary.

        This estimator, given X and D, optimises a functional of the
        following form:

        (1/2)||X - CD||_F^2 +  psi(C)

        where the norm of the reconstruction error is a Frobenious matrix norm
        and the function psi is a combination of penalties that act
        row-wise on both the matrices.
        It is possible to specify which kind of penalties you want to use on
        the matrices, if there must be a non-negativity constraint and if the
        atoms in the dictionary must have norm equal to one.

        Parameters
        ----------

        D: array_like or sparse matrix, shape= (n_atoms, n_features)
            The dictionary.

        penalty: a sub-class of Penalty class in penalty.py file, optional
            It is applied on the coefficients and it can be
            - L0Penalty
            - L1Penalty
            - L2Penalty
            - ElasticNetPenalty
            - GroupLassoPenalty
            - LInfPenalty

        non_negativity: bool, optional
            If true the coefficients are projected to the non-negative
            space.

        random_state: RandomState or int
            Seed to be use to initialise np.random.RandomState. If None each
            time RandomState is randomly initialised.

       """

    def __init__(self, D,  penalty=None, non_negativity=0, random_state=None):

        self.penalty = penalty
        self.non_negativity = non_negativity
        self.random_state = random_state
        self.X = None
        self.D = D
        self.C = None

    def fit(self, x, backtracking=0, n_iter=20000):

        """Function that fits the estimator on the matrices X and D.

        This function finds the coefficients with a proximal gradient
        algorithm. It computes the gradient on the reconstruction part of the
        functional and then applies the prox operator of the specified penalty.
        If the flat non-negative is set it project the rows of the matrices to
        the positive space.

        Parameters
        ----------

        x : array-like or sparse matrix shape =  (n_samples, n_features)
            The matrix to decompose.

        backtracking: bool, optional
            If True a procedure of backtracking is done on the step in order
            to avoid an increasing in the objective function.

        n_iter: int, optional
            Maximum number of iteration the minimization algorithm can
            perform.

        Returns
        -------
        self : object

        """

        # ______________ parameters control________________________________#
        x = check_array(x)
        self.D = check_array(self.D)
        self.X = x
        random_state = check_random_state(self.random_state)
        self.penalty = _check_penalty(self.penalty)

        logging.debug("Initializing coefficients..")

        # ________________optimization procedure____________________________#
        self.C = self._proximal_gradient_minimization(random_state,
                                                backtracking=backtracking,
                                                n_iter=n_iter)

        # __________________________final controls__________________________#
        logging.debug("Finished optimization")
        if self.non_negativity:
            assert (np.min(self.C) >= 0)

        return self

    def reconstruction_error(self):
        """
        Returns
        -------
        float:
            The reconstruction error for the current decomposition of the
            matrix. If no decomposition was run infinity is returned.
            A lower reconstruction error corresponds to a better approximation
            of the input data.
        """

        if self.X is None:
            return float("inf")
        return (np.linalg.norm(self.X - self.C.dot(self.D))/
                np.linalg.norm(self.X))

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
        decomposition is available NaN is returnes.

        Returns
        -------
        float:
            The value of the objective function.
        """
        if x is None:
            x = self.X
            if x is None:
                logging.info('You did not call the function fit. '
                             'Impossible to return an objective '
                             'function value')
                return float('NaN')

        if d is None:
            d = self.D
        if c is None:
            c = self.C

        reconstruction = np.linalg.norm(x - c.dot(d))**2
        penalty_coefficient = self.penalty.value(c)
        return reconstruction + penalty_coefficient

    def coefficients(self):
        """
        Returns
        -------

        C: array-like, shape = (n_samples, k)
            The learnt coefficients.

        """
        return self.C

    def score(self, *args):
        """
        Parameters
        ----------

        *args: optional
            Introduced for compatibility with sklearn GridSearchCV

        Returns
        -------
        float
            The score of the current decomposition. BIC value computed as
                 k(log(n_samples)-log(2\pi)) - 2*(||X - CD||)
            the highest is the score and better the decomposition is.
        """
        if self.X is None:
            return float("-inf")
        return - (np.log(self.X.shape[0])
                  + 2 * np.linalg.norm(self.X - self.C.dot(self.D)) ** 2)

    def _proximal_gradient_minimization(self, random_state, backtracking=0,
                                        n_iter=20000):
        x = self.X
        d = self.D
        n, p = x.shape
        k, _ = d.shape
        c = non_negative_projection(random_state.rand(n, k)*10-5,
                                    self.non_negativity)

        gamma = 1.1
        step = _step_lipschitz(d, gamma)
        iters = 20000
        epsilon = 1e-4
        objective = self.objective_function_value(d=d, c=c)

        c_old = c
        logging.debug("Starting optimization")
        for i in range(iters):
            gradient = (c.dot(d) - x).dot(d.T)

            if backtracking:
                c = self._update_with_backtracking(x, d, c, gradient, step)
            else:
                c = self._simple_update(c, gradient, step)

            new_objective = self.objective_function_value(d=d, c=c)
            difference_objective = new_objective - objective
            objective = new_objective
            difference_c = np.linalg.norm(c - c_old)
            c_old = c

            step = _step_lipschitz(d, gamma)

            logging.debug("Iteration: %i"%(i))
            logging.debug("Objective function: %f"%(objective))
            logging.debug("Difference between new objective and "
                          "previous one: %f"%(difference_objective))
            logging.debug("Difference between previous and new coefficients: "
                          "%f"%(difference_c))
            logging.debug("\n")

            assert ((not np.isnan(difference_objective)) and
                    (not np.isinf(difference_objective)) and
                    abs(difference_objective) < 1e+20)

            if (abs(difference_objective) <= epsilon and
                    difference_c <= epsilon):
                break

        return c

    def _simple_update(self, c, gradient, step):
        c = self.penalty. \
            apply_prox_operator(c - step * gradient, gamma=step)
        c = non_negative_projection(c, self.non_negativity)
        return c

    def _update_with_backtracking(self, x, d, c, gradient, step):
        c_0 = c
        error = 0.5 * np.linalg.norm(x - c_0.dot(d))**2
        sigma = 0.9
        for i in range(1, 1000):
            # compute new matrices
            c_1 = self.penalty. \
                apply_prox_operator(c_0 - step * gradient, gamma=step)
            c_1 = non_negative_projection(c_1, self.non_negativity)
            difference_c = np.linalg.norm(c_1 - c_0)

            first_part = self.objective_function_value(d=d, c=c_1)

            second_part = (error
                          + self.coeff_penalty.value(c_1)
                          + (difference_c*gradient).trace()
                          + step*sigma**i * difference_c)

            if first_part <= second_part:
                break

        return c_1


# ____________________________UTILITY FUNCTIONS_______________________________
def _step_lipschitz(d, gamma_c):
    step_c = max(0.0001, gamma_c * np.linalg.norm(d.T.dot(d)))
    return 1/step_c





