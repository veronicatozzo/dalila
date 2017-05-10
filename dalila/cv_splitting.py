from sklearn.model_selection._split import BaseShuffleSplit
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.utils import check_random_state
import numpy as np


class MonteCarloBootstrap(BaseShuffleSplit):
    """Random permutation cross-validator
    Yields indices to split data into training and test sets.
    Note: contrary to other cross-validation strategies, random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int (default 10)
        Number of re-shuffling & splitting iterations.
    test_size : float, int, or None, default 0.1
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples. If None,
        the value is automatically set to the complement of the train size.
    train_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Examples
    --------
    from sklearn.model_selection import ShuffleSplit
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([1, 2, 1, 2])
    rs = ShuffleSplit(n_splits=3, test_size=.25, random_state=0)
    rs.get_n_splits(X)

    print(rs)
    ShuffleSplit(n_splits=3, random_state=0, test_size=0.25, train_size=None)
    for train_index, test_index in rs.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...  # doctest: +ELLIPSIS
    TRAIN: [3 1 0] TEST: [2]
    TRAIN: [2 1 3] TEST: [0]
    TRAIN: [0 2 1] TEST: [3]
    rs = ShuffleSplit(n_splits=3, train_size=0.5, test_size=.25,
    ...                   random_state=0)
    for train_index, test_index in rs.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...  # doctest: +ELLIPSIS
    TRAIN: [3 1] TEST: [2]
    TRAIN: [2 1] TEST: [0]
    TRAIN: [0 2] TEST: [3]
    """

    def _iter_indices(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        n_train, n_test = _validate_shuffle_split(n_samples, self.test_size,
                                                  self.train_size)
        rng = check_random_state(self.random_state)
        for i in range(self.n_splits):
            # random partition
            ind_train = rng.randint(0, high=X.shape[0],
                                    size=n_train)
            ind_test = list(set(np.arange(0, X.shape[0])) -
                            set(np.unique(ind_train)))
            yield ind_train, ind_test
