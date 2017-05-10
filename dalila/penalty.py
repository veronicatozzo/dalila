from __future__ import print_function, division

import numpy as np
import sys
from itertools import product


class Penalty:
    """
    Class to represent a general penalty.
    """

    def apply_by_row(self, x, gamma):
        if x.ndim < 2:
            return self.prox_operator(x, gamma)
        else:
            new_x = np.zeros_like(x)
            for r in range(0, x.shape[0]):
                new_x[r, :] = self.prox_operator(x[r, :], gamma)
        return new_x

    def apply_by_col(self, x, gamma):
        if x.ndim < 2:
            return self.prox_operator(x, gamma)
        else:
            new_x = np.zeros_like(x)
            for c in range(0, x.shape[1]):
                new_x[:, c] = self.prox_operator(x[:, c], gamma)
            return new_x

    def apply_prox_operator(self, x, gamma):
        return x

    def prox_operator(self, x, gamma):
        return x

    def make_grid(self):
        return []

    def value(self, x):
        return 0


class L1Penalty(Penalty):

    """
    Class representing l1 penalty.

    The prox operator in case of 2D matrix will be applied rowwise.

    Parameters
    ----------

    _lambda: float
    The regularization value. Allawed values are between 0 and 1.

    """

    def __init__(self, _lambda):
       self._lambda = _lambda

    def apply_prox_operator(self, x, gamma):
        return self.apply_by_row(x, gamma)

    def prox_operator(self, x, gamma):
        sign = np.sign(x)
        np.maximum(np.abs(x) - (self._lambda*gamma), 0, out=x)
        x *= sign
        return x

    def make_grid(self, low=0.001, high=1, number=10):
        values = np.logspace(np.log10(low), np.log10(high), number)
        l = []
        for (i, v) in enumerate(values):
           l.append(L1Penalty(v))
        return l

    def __str__(self):
        return "L1Penalty(" + str(self._lambda) + ")"

    def value(self, x):
        return self._lambda * np.linalg.norm(x, 1)


class L2Penalty(Penalty):
    """
        Class representing l2 penalty.

        The prox operator in case of 2D matrix will be applied rowwise.

        Parameters
        ----------

        _lambda: float
        The regularization value. Allowed values are between 0 and 1.

        """

    def __init__(self, _lambda):
        if _lambda < 0 or _lambda > 1:
            print('\033[1;31m Wrong value for the l2 penalty.\033[1;m')
            sys.exit(0)

        self._lambda = _lambda

    def apply_prox_operator(self, x, gamma):
        return self.apply_by_row(x, gamma)

    def prox_operator(self, x, gamma):
        norm = np.linalg.norm(x) + 1e-10  # added constant for stability
        x *= max(1 - (gamma*self._lambda) / norm, 0)
        return x

    def make_grid(self, low=0.001, high=1, number=10):
        values = np.logspace(np.log10(low), np.log10(high), number)
        l = []
        for (i, v) in enumerate(values):
            l.append(L2Penalty(v))
        return l

    def __str__(self):
        return "L2Penalty(" + str(self._lambda) + ")"

    def value(self, x):
        return self._lambda * np.sum(np.apply_along_axis(np.linalg.norm,
                                                         axis=1, arr=x))


class ElasticNetPenalty(Penalty):
    """
        Class representing elastic net penalty.

        The prox operator in case of 2D matrix will be applied rowwise.

        Parameters
        ----------

        _lambda1: float
        The regularization value for the l1 penalty.

        _lambda2: float
        The regularization value for the l2 penalty.

        alpha: float
            alpha is the percentage of l1 regularization while (1-alpha) is the
            percentage of l2 regularization.

        """

    def __init__(self, _lambda1, _lambda2, alpha):
        self._lambda1 = _lambda1
        self._lambda2 = _lambda2
        self.alpha = alpha

    def apply_prox_operator(self, x, gamma):
        return self.apply_by_row(x, gamma)

    def prox_operator(self, x, gamma):
        sign = np.sign(x)
        np.maximum(np.abs(x) - (self._lambda * gamma), 0, out=x)
        x *= sign
        x /= (1. + self.alpha * self._lambda2 * gamma)
        return x

    def make_grid(self, low=0.001, high=1, number=5):
        a = np.logspace(np.log10(low), np.log10(high), number)
        b = np.logspace(np.log10(low), np.log10(high), number)
        alpha = np.arange(0.1, 1, 0.2)
        l1 = product(a, b)
        values = product(l1, alpha)
        l = []
        for (i, v) in enumerate(values):
            l.append(ElasticNetPenalty(v[0][0], v[0][1], v[1]))
        return l

    def __str__(self):
        return "ElasticNetPenalty( " \
               + str(self._lambda1) + ", " \
               + str(self._lambda2) + ", " + str(self.alpha) + ")"

    def value(self, x):
        l1 = self._lambda1*self.alpha * np.linalg.norm(x, 1)
        l2 = self._lambda2*(1-self.alpha) + \
             np.sum(np.apply_along_axis(np.linalg.norm, axis=1, arr=x))
        return l1 + l2


class L0Penalty(Penalty):
    """
        Class representing L0 penalty.

        The prox operator in case of 2D matrix will be applied rowwise.

        Parameters
        ----------

        _s: int
        The number of non-zero elements in each row.

        """

    def __init__(self, s):
        self.s = s

    def apply_prox_operator(self, x, gamma):
        return self.apply_by_row(x, gamma)

    def prox_operator(self, x, gamma):
        indices = np.argsort(x)
        x_new = np.zeros_like(x)
        x_new[indices[-self.s:]] = x[indices[-self.s:]]
        return x_new

    def make_grid(self, low=0, high=1, number=5):
        values = np.linspace(low, high, number)
        l = [L0Penalty(v) for v in values]
        return l

    def __str__(self):
        return "L0Penalty( " \
               + str(self.s) + ")"

    def value(self, x):
        return self.s

