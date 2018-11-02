from __future__ import print_function, division

import logging

import numpy as np
from itertools import product, combinations, chain
import bintrees


class Penalty:
    """
    Super class that represents a general empty penalty.
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
    The regularization value. Allawed values are higher than 0.

    """

    def __init__(self, _lambda):
       self._lambda = _lambda

    def apply_prox_operator(self, x, gamma):
        if self._lambda < 0:
            logging.error("A negative regularization parameter was used")
            raise ValueError("A negative regularization parameter was used")

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
        The regularization value. Allowed values are higher than 0.

        """

    def __init__(self, _lambda):
         self._lambda = _lambda

    def apply_prox_operator(self, x, gamma):
        if self._lambda < 0:
            logging.error("A negative regularisation parameter was used")
            raise ValueError("A negative regularization parameter was used")

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
        if self._lambda1 < 0 or self._lambda2 < 0:
            logging.error("A negative regularisation parameter was used")
            raise ValueError("A negative regularization parameter was used")
        if self.alpha < 0 or self.alpha>1:
            logging.error("The alpha value of elastic net penalty has to be "
                          "in the interval [0,1]")
            raise ValueError("The alpha value of elastic net penalty has to be "
                          "in the interval [0,1]")

        return self.apply_by_row(x, gamma)

    def prox_operator(self, x, gamma):
        sign = np.sign(x)
        np.maximum(np.abs(x) - (self._lambda1 * gamma), 0, out=x)
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
        if self.s < 0:
            logging.error("A negative regularisation parameter was used")
            raise ValueError("A negative regularization parameter was used")
        if self.s > x.shape[1]:
            logging.error("The number of non-zero elements to impose with L0 "
                          "penalty cannot be higher than the number of "
                          "features")
            raise ValueError("The number of non-zero elements to impose with "
                             "L0 penalty cannot be higher than the number of "
                             "features")
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


class GroupLassoPenalty(Penalty):
    """
        Class representing non-overlapping Group Lasso penalty.

        The prox operator in case of 2D matrix will be applied rowwise.

        Parameters
        ----------

        _groups: list of int
        List of the groups in which subdivide the considered feature space.
        There cannot be overlaps.

        _lambda: float
        The regularization value. Allawed values are higher than 0.

        """

    def __init__(self, _groups, _lambda):
        self._groups = _groups
        self._lambda = _lambda

    def apply_prox_operator(self, x, gamma):
        if self._lambda < 0:
            logging.error("A negative regularisation parameter was used")
            raise ValueError("A negative regularization parameter was used")

        l = list(set().union(*self._groups))
        if not (l == list(np.arange(x.shape[1]))):
            logging.error("The groups in group lasso must cover all the "
                          "features")
            raise ValueError("The groups in group lasso must cover all the "
                          "features")

        for pair in combinations(self._groups, r=2):
            if len(set(pair[0]) & set(pair[1])) > 0:
                logging.error("There are overlapping groups")
                raise ValueError("There are overlapping groups")

        new_x = np.zeros_like(x)
        for r in range(0, x.shape[0]):
            for g in self._groups:
                new_x[r, g] = self.prox_operator(x[r, g], gamma)
        return new_x

    def prox_operator(self, x, gamma):
        if np.linalg.norm(x) < self._lambda*gamma:
            return np.zeros_like(x)
        x *= (1 - (self._lambda*gamma)/np.linalg.norm(x))
        return x

    def make_grid(self, low=0, high=1, number=5):
        values = np.linspace(low, high, number)
        l = [GroupLassoPenalty(self._groups, v) for v in values]
        return l

    def __str__(self):
        return "GroupLassoPenalty( " \
               + str(self._groups) + str(self._lambda) + ")"

    def value(self, x):
        res = 0
        for r in range(x.shape[0]):
            for g in self._groups:
                res += np.linalg.norm(x[r, g])
        return self._lambda * res


class LInfPenalty(Penalty):

    def __init__(self, _lambda):
        self._lambda = _lambda

    def apply_prox_operator(self, x, gamma):
        if self._lambda < 0:
            logging.error("A negative regularisation parameter was used")
            raise ValueError("A negative regularization parameter was used")
        return self.apply_by_col(x, gamma)

    def prox_operator(self, x, gamma):
        # norm = np.linalg.norm(x) + 1e-10  # added constant for stability
        # x *= max(1 - (gamma*self._lambda) / norm, 0)
        x -= gamma*self._lambda * \
             self._projection_L1ball(x/(gamma*self._lambda))
        return x

    def make_grid(self, low=-3, high=1, number=10):
        values = np.logspace(low, high, number)
        l = []
        for (i, v) in enumerate(values):
            l.append(LInfPenalty(v))
        return l

    def projection_L1ball(self, v):

        """Find the projection of a vector onto the L1 ball. See for reference
        'Efficient projections onto L1-ball for learning in high dimensions'
        Duchi , Schwartz, Singer. SECTION 4, Figure 2.
        Parameters
        -------------------
        v : vector to be projected
        Returns
        -------------------
        w : projection on v onto the L1 ball of radius z
        """

        vector_copy = np.abs(v)
        keys_ar = np.unique(vector_copy)
        r = map(lambda k: (np.where(k == vector_copy)[0]).sum(), keys_ar)
        r = list(r)
        # r = np.where(j == vector_copy)[0].sum()
        # print("where ", r)
        # r = map(lambda k: len(np.where(k == vector_copy)[0]), keys_ar)
        # print("where2 ", r)
        n_elements = np.cumsum(r[::-1])[::-1]  # r di cui abbiamo bisogno
        s_sum = keys_ar * r
        cum_sum = np.cumsum(s_sum[::-1])[::-1]

        dictionary = {}
        for i in range(len(keys_ar)):
            dictionary[keys_ar[i]] = [n_elements[i], cum_sum[i]]

        rb_tree = bintrees.RBTree(dictionary)
        theta = self._pivotsearch(rb_tree, rb_tree._root, 0., 0.)

        return np.clip(vector_copy - theta, a_min=0, a_max=np.inf) * np.sign(v)

    def _pivotsearch(self, rb_tree, v, rho, s, v_star=np.inf, rho_star=0.,
                     s_star=0.):
        z = 1.  # ray of the ball, 1
        rho_hat = v.value[0]
        s_hat = v.value[1]

        if s_hat < v.key * rho_hat + z:
            if v_star > v.key:
                v_star = v.key
                rho_star = rho_hat
                s_star = s_hat
            if self._is_leaf(v):
                return (s_star - z) / rho_star
            if v.left is not None:
                del rb_tree[v.key:]
                return self._pivotsearch(rb_tree, rb_tree._root, rho_hat,
                                         s_hat, v_star, rho_star,
                                         s_star)  # node v.left
            else:
                return (s_star - z) / rho_star  # "no left child"

        else:
            if self._is_leaf(v):
                return (s_star - z) / rho_star
            if v.right is not None:
                del rb_tree[:v.key]
                del rb_tree[v.key]
                return self._pivotsearch(rb_tree, rb_tree._root, rho, s,
                                         v_star, rho_star,
                                         s_star)  # node v.right
            else:
                return (s_star - z) / rho_star  # "no right child"

    def _is_leaf(self, v):
        return v.left is None and v.right is None


