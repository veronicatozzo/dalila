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
    """ An estimator for dictionary learning based on alternating prox methods.

        This estimator optimises a functional of the following form:

        (1/2)||X - CD||_F^2 + phi(D) + psi(C)

        where the norm of the reconstruction error is a Frobenious matrix norm
        and the functions phi and psi are a combination of penalties that act
        row-wise on both the matrices.
        It is possible to specify which kind of penalties you want to use on
        the matrices, if there must be a non-negativity constraint and if the
        atoms in the dictionary must have norm equal to one.

        Parameters
        ----------

        k: int
            Number of atoms in which decompose the input matrix.

        dict_penalty : a sub-class of Penalty in penalty.py file, optional
            It is applied on the dictionary and it can be L0Penalty,
            L1Penalty, L2Penalty, ElasticNetPenalty

        coeff_penalty: a sub-class of Penalty class in penalty.py file,
            optional
            It is applied on the coefficients and it can be L0Penalty,
            L1Penalty, L2Penalty, ElasticNetPenalty

        dict_normalization: int, optional
            If different than zero the atoms are normalized to have l2-norm
            equal to 1.

        non_negativity: string, optional
            If 'none' (default) the atoms and the coefficients can be both
            non negative.
            If 'both' non-negativity is applied on both matrices.
            if 'coeff' non-negativity is applied only on the matrix of the
            coefficients.

        random_state: RandomState or int
            Seed to be use to initialise np.random.RandomState. If None each
            time RandomState is randomly initialised.

       """

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

        """Function that fits the estimator on the matrix X.

        This function finds the decomposition in dictionary and coefficients
        with an alternating proximal gradient algorithm. It iteratively
        computes the gradient on one of the two matrices by keeping the other
        fixed and then apply the prox operator of the specified penalty.
        If the flat non-negative is set it project the rows of the matrices to
        the positive space, while if the dict_normalization flag is set the
        rows of the dictionaries are normalised to have l2-norm equal to 1.

        Parameters
        ----------

        x : array-like or sparse matrix shape =  (n_samples, n_features)
            The matrix to decompose.

        y:
            Inserted for compatibility with sklearn library.

        n_iter: int, optional
            Maximum number of iterations the algorithm does before stopping.

        backtracking: bool, optional
            If True a procedure of backtracking is done on the step in order
            to avoid an increasing in the objective function.

        Returns
        -------
        self : object

        """

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
        return (np.linalg.norm(self.X - self.C.dot(self.D)) /
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

    def decomposition(self):
        """
        Returns
        -------

        C: array-like, shape = (n_samples, k)
            The learnt coefficients.

        D: array_like, shape = (k, n_features)
            The learnt dictionary.

        """
        return self.C, self.D

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
            -log(k)*log(n_samples) - 2.3*(self.objective_function_value())
            the highest is the score and better the decomposition is.
        """

        if self.X is None:
            return float("-inf")
        return - (np.log(self.X.shape[0])*self.k \
                    + 2 * np.linalg.norm(self.X - self.C.dot(self.D))**2)

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

    def _update_with_backtracking(self, d, c, gradient_d, gradient_c,
                                  step_d, step_c):
        x = self.X
        d_0 = d
        c_0 = c
        error = 0.5 * np.linalg.norm(x - c_0.dot(d_0))**2
        sigma = 0.9
        for i in range(1, 1000):
            # compute new matrices
            d_1 = self.dict_penalty. \
                apply_prox_operator(d_0 - step_d * gradient_d, gamma=step_d)
            d_1 = non_negative_projection(d_1, self.non_negativity, 'dict')
            difference_d = np.linalg.norm(d_1 - d_0)

            c_1 = self.coeff_penalty. \
                apply_prox_operator(c_0 - step_c * gradient_c, gamma=step_c)
            c_1 = non_negative_projection(c_1, self.non_negativity, 'coeff')
            difference_c = np.linalg.norm(c_1 - c_0)

            first_part = self.objective_function_value(d=d_1, c=c_1)

            second_part = (error +
                           self.dict_penalty.value(d_1) +
                           self.coeff_penalty.value(c_1) +
                           (difference_c*gradient_c).trace() +
                           (difference_d*gradient_d).trace() +
                           step_c*sigma**i * difference_c +
                           step_d*sigma**i * difference_d)

            if first_part <= second_part:
                break

        return d_1, c_1


###############################################################################
class StabilityDictionaryLearning(DictionaryLearning):
    """ Estimator for dictionary learning (DL) based on prox methods and clusters.

        This estimator optimises a functional of the following form:

        (1/2)||X - CD||_F^2 + phi(D) + psi(C)

        where the norm of the reconstruction error is a Frobenious matrix norm
        and the functions phi and psi are a combination of penalties that act
        row-wise on both the matrices.
        It is possible to specify which kind of penalties you want to use on
        the matrices, if there must be a non-negativity constraint and if the
        atoms in the dictionary must have norm equal to one.

        The algorithm performs many decomposition using DL and after each
        iteration it clusters the atoms obtained in every iteration in order to
        have final atoms that are stable w.r.t to data noise.

        For more details see Alexandrov et al, Cell 2013.

        Parameters
        ----------

        k: int
            Number of atoms in which decompose the input matrix.

        dict_penalty : a sub-class of Penalty in penalty.py file,
            optional
            It is applied on the dictionary and it can be L0Penalty,
            L1Penalty, L2Penalty, ElasticNetPenalty

        coeff_penalty: sa sub-class of Penalty class in penalty.py file,
            optional
            It is applied on the coefficients and it can be L0Penalty,
            L1Penalty, L2Penalty, ElasticNetPenalty

        dict_normalization: int, optional
            If different than zero the atoms are normalized to have l2-norm
            equal to 1.

        non_negativity: string, optional
            If 'none' (default) the atoms and the coefficients can be both
            non negative.
            If 'both' non-negativity is applied on both matrices.
            if 'coeff' non-negativity is applied only on the matrix of the
            coefficients.


        random_state: RandomState or int
            Seed to be use to initialise np.random.RandomState. If None each
            time RandomState is randomly initialised.

       """

    def __init__(self, k, dict_penalty=None, coeff_penalty=None,
                 dict_normalization=1, non_negativity='none',
                 random_state=None):
        super(StabilityDictionaryLearning, self)\
            .__init__(k, dict_penalty, coeff_penalty, 1, non_negativity,
                      random_state)
        self.meanD = None
        self.meanC = None
        self.stability = None
        self.Ds_sequence = None
        self.Cs_sequence = None

    def fit(self, x, y=None, backtracking=0, n_iter=20000, epsilon=1e-4):
        """Function that fits the estimator on the matrix X.

           This function finds the decomposition in dictionary and coefficients
           with an alternating proximal gradient algorithm. It iteratively
           computes the gradient on one of the two matrices by keeping the
           other fixed and then apply the prox operator of the specified
           penalty. If the flat non-negative is set it project the rows of the
           matrices to the positive space, while if the dict_normalization flag
           is set the rows of the dictionaries are normalised to have l2-norm
           equal to 1.

           Parameters
           ----------

           x : array-like or sparse matrix shape =  (n_samples, n_features)
               The matrix to decompose.

           y:
               Inserted for compatibility with sklearn library.

           n_iter: int, optional
               Maximum number of iterations the algorithm does before stopping.

           backtracking: bool, optional
               If True a procedure of backtracking is done on the step in order
               to avoid an increasing in the objective function.

           epsilon: float, optional
                The difference between the each iteration of DL at which the
                algorithm stops to return the solution.

           Returns
           -------
           self : object

           """
        x = check_array(x)
        self.X = x
        n, p = x.shape
        random_state = check_random_state(self.random_state)
        self.dict_penalty = _check_penalty(self.dict_penalty)
        self.coeff_penalty = _check_penalty(self.coeff_penalty)
        _check_number_of_atoms(self.k, p, n)
        _check_non_negativity(self.non_negativity, x)

        difference = 10
        n, d = x.shape

        D = random_state.rand(self.k, d)
        Ds = []
        Cs = []

        mean_D = D
        # until convergence
        while difference > epsilon:
            old_mean_D = mean_D
            # random bootstrap with replacement and fit
            samples = _sampling(x, random_state)
            super(StabilityDictionaryLearning, self)\
                .fit(samples, n_iter=n_iter, backtracking=backtracking)
            Ds.append(self.D)
            Cs.append(self.C)

            Ds, Cs, mean_D, mean_C, stability = \
                _compute_clusters_and_silhouettes(Ds, Cs)

            # normalization
            for i in range(mean_D.shape[0]):
                mean_D[i, :] = mean_D[i, :] / np.sum(mean_D[i, :])

            difference = np.sum((old_mean_D - mean_D) ** 2)
            logging.debug("Difference " + str(difference))

        self.meanD = mean_D
        self.meanC = mean_C
        self.stability = stability
        self.Ds_sequence = Ds
        self.Cs_sequence = Cs
        return self

    def decomposition(self):
        """
            Returns
            -------

            C: array-like, shape = (n_samples, k)
                The learnt mean coefficients.

            D: array_like, shape = (k, n_features)
                The learnt mean dictionary, the centroids of the clusters of
                atoms.

            """
        return self.meanC, self.meanD

    def stability(self):
        """

        Returns
        -------
        float
            The mean stability obtained with the clusterization. Where for
            stability we mean the average silhouette over all the clusters.
        """
        return self.stability


# ____________________________UTILITY FUNCTIONS_______________________________
def _step_lipschitz(d, c, gamma_d,  gamma_c):
    step_c = max(0.0001, gamma_c * np.linalg.norm(d.T.dot(d)))
    step_d = max(0.0001, gamma_d * np.linalg.norm(c.dot(c.T)))
    return (1/step_d), (1/step_c)


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


