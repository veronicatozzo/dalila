from __future__ import division

import sys
import logging

from sklearn.externals.joblib import Parallel, parallel_backend, \
    register_parallel_backend
from distributed.joblib import DistributedBackend

from sklearn.model_selection import GridSearchCV, ShuffleSplit

from sklearn.utils import check_array
from multiprocessing import cpu_count

from dalila.dictionary_learning import DictionaryLearning
from dalila.utils import check_non_negativity
from dalila.cv_splitting import MonteCarloBootstrap

def tune_parameters(X, estimator=None, analysis=0, non_negative="none",
                    distributed=0, scheduler_host="",
                    max_k=None,
                    dict_penalty_range=(0.0001, 1, 10),
                    coeff_penalty_range=(0.0001, 1, 10)):
 #TODO: aggiungere parametri nella descrizione
    """

    Parameters
    ----------
    X: array-like, shape=(n_samples, n_features)
        The matrix to decompose and analyse.

    estimator: DictionaryLearningEstimator, optional
        The estimator you want to use to analyse the matrix. If None only the
        research on the best number of atoms will be done.

    analysis: int, optional
        The type of tuning you want to perform.
        - 0: tune together number of atoms and dictionary penalty and then the
             coefficients penalty
        - 1: tune only the penalties and take the number of atoms as specified in
             the estimator
        - 2: tune only the number of atoms
        - 3: tune all together, number of atoms and penalties

    non_negative: string, optional
        If "none" no negativity is imposed on the decomposition, if "coeff" only
        negativity on the coefficient is imposed. If "both" negativiy is on both
        decomposition matrices.

    dict_penalty_range: float tuple, optional (low, high, number)
        It gives the interval in which tune the dictionary penalty and the
        number of values to try.

    coeff_penalty_range: float tuple, optional (low, high, number)
        It gives the interval in which tune the coefficient penalty and the
        number of values to try.

    Returns
    -------
    DictionaryLearning
    The best estimator found with the cross-validation procedure.

    """
    _check_estimator(estimator)
    _check_range(dict_penalty_range)
    _check_range(coeff_penalty_range)
    check_non_negativity(non_negative, X)
    if estimator is None:
        analysis = 2
    else:
        estimator.non_negativity = non_negative

    X = check_array(X)
    n, p = X.shape

    if max_k is None:
        max_k = int(min(p, 0.75 * n) / 2)  # generally the optimal
                                           # number of k is low

    if (analysis in [0, 1, 3] and
       (dict_penalty_range is None or coeff_penalty_range is None)):
        logging.error("The range cannot be None")
    if analysis == 0:
        return _find_everything_sequentially(X, estimator, max_k,
                                             dict_penalty_range,
                                             coeff_penalty_range,
                                             distributed, scheduler_host)
    elif analysis == 1:
        return _find_penalties(X, estimator, dict_penalty_range,
                               coeff_penalty_range,
                               distributed, scheduler_host)
    elif analysis == 2:
        return _find_number_of_atoms(X, max_k, non_negative,
                                     distributed, scheduler_host)
    elif analysis == 3:
        return _find_everything(X, estimator, max_k, dict_penalty_range,
                                coeff_penalty_range,
                                distributed, scheduler_host)
    else:
        logging.error("Unknown type of research, please try with another "
                      "setting")


def _find_everything_sequentially(x, estimator, max_k,
                                  dict_penalty_range, coeff_penalty_range,
                                  distributed=0, scheduler_host=""):

    ss = MonteCarloBootstrap(n_splits=3, test_size=0.1)

    # ---------------------first part
    params_dict = _get_params_dict(estimator,
                                   dict_penalty_range=dict_penalty_range,
                                   coeff_penalty_range=None)
    params_dict['k'] = list(range(2, max_k))
    logging.debug("Starting cross validation...")
    gscv = GridSearchCV(estimator, params_dict, cv=ss,
                        iid=True, refit=True, verbose=1)
    if distributed:
        register_parallel_backend('distributed', DistributedBackend)

        with parallel_backend('distributed',
                              scheduler_host=scheduler_host):
            gscv.fit(x)
    else:
        gscv.fit(x)
    best_est_dict = gscv.best_estimator_

    # ---------------------second part
    params = _get_params_dict(estimator, None, coeff_penalty_range)
    gscv = GridSearchCV(best_est_dict, params, cv=ss,
                        n_jobs=cpu_count() - 2, iid=True, refit=True,
                        verbose=1)
    if distributed:
        register_parallel_backend('distributed', DistributedBackend)

        with parallel_backend('distributed',
                              scheduler_host=scheduler_host):
            gscv.fit(x)
    else:
        gscv.fit(x)

    return gscv.best_estimator_


def _find_number_of_atoms(x, max_k, non_negative='none', distributed=0,
                          scheduler_host=""):
    estimator = DictionaryLearning(k=0,
                                   non_negativity=non_negative,
                                   random_state=None)
    params = {'k': list(range(2, max_k))}
    ss = MonteCarloBootstrap(n_splits=3, test_size=0.1)

    gscv = GridSearchCV(estimator, params, cv=ss,
                      n_jobs=cpu_count()-2, iid=True,
                      refit=True,
                      verbose=1)

    if distributed:
        register_parallel_backend('distributed', DistributedBackend)

        with parallel_backend('distributed',
                              scheduler_host=scheduler_host):
            gscv.fit(x)
    else:
        gscv.fit(x)
    print("Number of atoms found: " +
          gscv.cv_results_['params'][gscv.best_index_]['k'])
    return gscv.best_estimator_


def _find_penalties(x, estimator,
                    dict_penalty_range=(0.001, 1, 10),
                    coeff_penalty_range=(0.001, 1, 10),
                    distributed=0, scheduler_host=""):

    ss = MonteCarloBootstrap(n_splits=3, test_size=0.1)
    params = _get_params_dict(estimator, dict_penalty_range,
                              coeff_penalty_range)
    gscv = GridSearchCV(estimator, params, cv=ss, n_jobs=(cpu_count()-5),
                        iid=True,  refit=True, verbose=1)
    if distributed:
        register_parallel_backend('distributed', DistributedBackend)

        with parallel_backend('distributed',
                              scheduler_host=scheduler_host):
            gscv.fit(x)
    else:
        gscv.fit(x)

    return gscv.best_estimator_


def _find_everything(x, estimator, max_k,
                     dict_penalty_range=(0.001, 1, 10),
                     coeff_penalty_range=(0.001, 1, 10),
                     distributed=0, scheduler_host=""):

    params = _get_params_dict(estimator, dict_penalty_range,
                              coeff_penalty_range)

    ss = MonteCarloBootstrap(n_splits=3, test_size=0.1)
    params['k'] = list(range(2, max_k))
    gscv = GridSearchCV(estimator, params, cv=ss,
                        n_jobs=cpu_count() - 2, iid=True, refit=True,
                        verbose=1)

    if distributed:
        register_parallel_backend('distributed', DistributedBackend)

        with parallel_backend('distributed',
                              scheduler_host=scheduler_host):
            gscv.fit(x)
    else:
        gscv.fit(x)

    return gscv.best_estimator_


def _get_params_dict(estimator, dict_penalty_range, coeff_penalty_range):

    if estimator.dict_penalty is None and estimator.coeff_penalty is None:
        return {}

    if estimator.dict_penalty is None:
        if coeff_penalty_range is None:
            return {}
        else:
            return {'coeff_penalty': (estimator.coeff_penalty.
                                      make_grid(coeff_penalty_range[0],
                                                coeff_penalty_range[1],
                                                coeff_penalty_range[2]))}
    if estimator.coeff_penalty is None:
        if dict_penalty_range is None:
            return {}
        else:
            return {'dict_penalty': (estimator.dict_penalty.
                                     make_grid(dict_penalty_range[0],
                                               dict_penalty_range[1],
                                               dict_penalty_range[2]))}
    if coeff_penalty_range is None:
        return {'dict_penalty': (estimator.dict_penalty.
                                 make_grid(dict_penalty_range[0],
                                           dict_penalty_range[1],
                                           dict_penalty_range[2]))}
    elif dict_penalty_range is None:
        return {'coeff_penalty': (estimator.coeff_penalty.
                                  make_grid(coeff_penalty_range[0],
                                            coeff_penalty_range[1],
                                            coeff_penalty_range[2]))}
    else:
        return {'dict_penalty': (estimator.dict_penalty.
                                 make_grid(dict_penalty_range[0],
                                           dict_penalty_range[1],
                                           dict_penalty_range[2])),
                'coeff_penalty': (estimator.coeff_penalty.
                                  make_grid(coeff_penalty_range[0],
                                            coeff_penalty_range[1],
                                            coeff_penalty_range[2]))}


def _check_estimator(estimator):
    if not (isinstance(estimator, DictionaryLearning)):
        logging.error('Unknown estimator for the '
                      'dictionary learning optimization.')
        sys.exit(0)


def _check_range(r):
    if r is None:
        return
    if not len(r) == 3:
        logging.error('Too few elements in range specification.')
        sys.exit(0)

    if r[0] < 0:
        logging.error('The minimum number for the range is 0.')
        sys.exit(0)

    if r[2] > 10:
        logging.warning('Using more than ten values for each penalty may '
                        'require a long computational time.')

