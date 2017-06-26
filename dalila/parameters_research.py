from __future__ import division

import logging

from sklearn.externals.joblib import Parallel, parallel_backend, \
    register_parallel_backend
from distributed.joblib import DistributedBackend
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.utils import check_array, check_random_state
from multiprocessing import cpu_count

from dalila.dictionary_learning import DictionaryLearning
from dalila.representation_learning import RepresentationLearning
from dalila.utils import _check_non_negativity, MonteCarloBootstrap


def tune_parameters_DL(X, estimator=None, analysis=3, non_negative="none",
                    distributed=0, scheduler_host="", max_k=None,
                    dict_penalty_range=(0.0001, 1, 10),
                    coeff_penalty_range=(0.0001, 1, 10), random_state=None):
    """
    Parameters tuner.

    It tunes the parameters of a dictionary learning estimator using 3-splits
    monte carlo sampling cross validation.

    Parameters
    ----------
    X: array-like, shape=(n_samples, n_features)
        The matrix to decompose and analyse.

    estimator: DictionaryLearning class, optional
        The estimator you want to use to analyse the matrix. If None only the
        research on the best number of atoms will be done.

    analysis: int, optional
        The type of tuning you want to perform.
        - 0: tune together number of atoms and dictionary penalty and then the
             coefficients penalty
        - 1: tune only the penalties and take the number of atoms as specified
             in the estimator
        - 2: tune only the number of atoms
        - 3: tune all together, number of atoms and penalties

    non_negative: string, optional
        If "none" no negativity is imposed on the decomposition, if "coeff"
        only negativity on the coefficient is imposed. If "both" negativiy is
        on both decomposition matrices.

    distributed: int, optional
        If 0 the parameters research will be executed in parallel on the
        computer the script is launched.
        If 1 the parameters research will be executed sequentially.
        If 2 the parameters research will be distributed on multiple machines
        connected by dask. In this case also scheduler_host must be speficied.

    scheduler_host: string, optional
        If distributed=2 it is necessary to specify the scheduler of the dask
        network. The string must be "ip_address:port", for example:
        "10.251.61.226:8786"

    max_k = int, optional
        The maximum number of atoms to try when you search for the right k.
        If None max_k will be computed as int(min(p, 0.75 * n) / 2)

    dict_penalty_range: float tuple, optional (low, high, number)
        It gives the interval in which tune the dictionary penalty and the
        number of values to try.

    coeff_penalty_range: float tuple, optional (low, high, number)
        It gives the interval in which tune the coefficient penalty and the
        number of values to try.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    GridSearchCV
    The resulting GridSearch.

    """

    # ------------------parameters control ---------------------------------- #
    X = check_array(X)
    random_state = check_random_state(random_state)
    _check_range(dict_penalty_range)
    _check_range(coeff_penalty_range)
    _check_non_negativity(non_negative, X)

    if estimator is None:
        analysis = 2
    else:
        _check_estimator(estimator)
        if estimator.non_negativity == "none":
            estimator.non_negativity = non_negative

    n, p = X.shape
    if max_k is None:
        max_k = int(min(p, 0.75 * n) / 2)  # generally the optimal
                                           # number of k is low

    if (analysis in [0, 1, 3] and
       (dict_penalty_range is None or coeff_penalty_range is None)):
        logging.ERROR("The range cannot be None")
        sys.exit(0)

    if distributed == 2:
        if scheduler_host is None:
            logging.ERROR("Distributed execution requires a scheduler "
                          "specification. Changing the type to parallel.")
            distributed = 1
        distributed = _check_scheduler(scheduler_host)

    if analysis == 0:
        return _find_everything_sequentially(X, estimator, max_k,
                                             dict_penalty_range,
                                             coeff_penalty_range,
                                             distributed, scheduler_host,
                                             random_state)
    elif analysis == 1:
        return _find_penalties(X, estimator, dict_penalty_range,
                               coeff_penalty_range,
                               distributed, scheduler_host,
                               random_state)
    elif analysis == 2:
        return _find_number_of_atoms(X, max_k, non_negative,
                                     distributed, scheduler_host,
                                     random_state)
    elif analysis == 3:
        return _find_everything(X, estimator, max_k, dict_penalty_range,
                                coeff_penalty_range,
                                distributed, scheduler_host,
                                random_state)
    else:
        logging.error("Unknown type of research, please try with another "
                      "setting")
        raise ValueError("Unkown type of research, please try with another"
                         "setting")


def tune_parameters_RL(X, D, estimator, non_negative=0,  distributed=0,
                       scheduler_host="", coeff_penalty_range=(0.0001, 1, 10),
                       random_state=None):
    """
    Parameters tuner.

    It tunes the parameters of a representations learning estimator using
    3-splits monte carlo sampling cross validation.

    Parameters
    ----------
    X: array-like, shape=(n_samples, n_features)
        The matrix to decompose and analyse.

    D: array-like, shape=(n_atoms, n_features)
        The dictionary.

    estimator: RepresentationLearning class, optional
        The estimator you want to use to analyse the matrix.

    non_negative: boolean, optional

    distributed: int, optional
        If 0 the parameters research will be executed in parallel on the
        computer the script is launched.
        If 1 the parameters research will be executed sequentially.
        If 2 the parameters research will be distributed on multiple machines
        connected by dask. In this case also scheduler_host must be speficied.

    scheduler_host: string, optional
        If distributed=2 it is necessary to specify the scheduler of the dask
        network. The string must be "ip_address:port", for example:
        "10.251.61.226:8786"

    coeff_penalty_range: float tuple, optional (low, high, number)
        It gives the interval in which tune the coefficient penalty and the
        number of values to try.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    GridSearchCV
    The resulting GridSearch.

    """

    # ------------------parameters control ---------------------------------- #
    X = check_array(X)
    D = check_array(D)
    random_state = check_random_state(random_state)
    _check_range(coeff_penalty_range)
    if estimator is None:
        logging.error("passed estimator was None")
        raise ValueError("passed estimator was None")
    _check_estimator(estimator)


    estimator.non_negativity = non_negative

    if distributed == 2:
        if scheduler_host is None:
            logging.ERROR("Distributed execution requires a scheduler "
                          "specification. Changing the type to parallel.")
            distributed = 1
        distributed = _check_scheduler(scheduler_host)

        ss = MonteCarloBootstrap(n_splits=3, test_size=0.1,
                                 random_state=random_state)

    params = _get_params_coeff(estimator, coeff_penalty_range,
                               representation_learning=1)

    jobs = 1 if distributed == 1 else cpu_count()
    gscv = GridSearchCV(estimator, params, cv=ss, n_jobs=(cpu_count() - 5),
                        iid=True, refit=True, verbose=1)
    if distributed == 2:
        register_parallel_backend('distributed', DistributedBackend)
        with parallel_backend('distributed',
                              scheduler_host=scheduler_host):
            gscv.fit(x)
    else:
        gscv.fit(x)

    return gscv


def _find_everything_sequentially(x, estimator, max_k,
                                  dict_penalty_range, coeff_penalty_range,
                                  distributed=0, scheduler_host="",
                                  random_state=None):

    ss = MonteCarloBootstrap(n_splits=3, test_size=0.1,
                             random_state=random_state)

    # ---------------------first part
    params= _get_params_dict(estimator,
                             dict_penalty_range=dict_penalty_range)
    params['k'] = list(range(2, max_k))

    jobs = 1 if distributed == 1 else cpu_count()
    gscv = GridSearchCV(estimator, params, cv=ss, n_jobs=jobs,
                        iid=True, refit=True, verbose=1)
    if distributed == 2:
        register_parallel_backend('distributed', DistributedBackend)
        with parallel_backend('distributed', scheduler_host=scheduler_host):
            gscv.fit(x)
    else:
        gscv.fit(x)
    best_est_dict = gscv.best_estimator_

    # ---------------------second part
    params = _get_params_coeff(estimator, coeff_penalty_range)
    gscv = GridSearchCV(best_est_dict, params, cv=ss,
                        n_jobs=cpu_count() - 2, iid=True, refit=True,
                        verbose=1)
    if distributed==2:
        register_parallel_backend('distributed', DistributedBackend)

        with parallel_backend('distributed',
                              scheduler_host=scheduler_host):
            gscv.fit(x)
    else:
        gscv.fit(x)

    return gscv


def _find_number_of_atoms(x, max_k, non_negative='none', distributed=0,
                          scheduler_host="", random_state=None):
    estimator = DictionaryLearning(k=0,
                                   non_negativity=non_negative,
                                   random_state=None)
    params = {'k': list(range(2, max_k))}
    ss = MonteCarloBootstrap(n_splits=3, test_size=0.1,
                             random_state=random_state)

    jobs = 1 if distributed == 1 else cpu_count()
    gscv = GridSearchCV(estimator, params, cv=ss,
                      n_jobs=cpu_count()-2, iid=True,
                      refit=True,
                      verbose=1)

    if distributed == 2:
        register_parallel_backend('distributed', DistributedBackend)
        with parallel_backend('distributed',
                              scheduler_host=scheduler_host):
            gscv.fit(x)
    else:
        gscv.fit(x)
    return gscv


def _find_penalties(x, estimator,
                    dict_penalty_range=(0.001, 1, 10),
                    coeff_penalty_range=(0.001, 1, 10),
                    distributed=0, scheduler_host="", random_state=None):

    ss = MonteCarloBootstrap(n_splits=3, test_size=0.1,
                             random_state=random_state)
    params = _get_params(estimator, dict_penalty_range, coeff_penalty_range)

    jobs = 1 if distributed == 1 else cpu_count()
    gscv = GridSearchCV(estimator, params, cv=ss, n_jobs=(cpu_count()-5),
                        iid=True,  refit=True, verbose=1)
    if distributed == 2:
        register_parallel_backend('distributed', DistributedBackend)
        with parallel_backend('distributed',
                              scheduler_host=scheduler_host):
            gscv.fit(x)
    else:
        gscv.fit(x)

    return gscv


def _find_everything(x, estimator, max_k,
                     dict_penalty_range=(0.001, 1, 10),
                     coeff_penalty_range=(0.001, 1, 10),
                     distributed=0, scheduler_host="",
                     random_state=None):

    params = _get_params(estimator, dict_penalty_range,
                              coeff_penalty_range)

    ss = MonteCarloBootstrap(n_splits=3, test_size=0.1,
                             random_state=random_state)
    params['k'] = list(range(2, max_k))

    jobs = 1 if distributed == 1 else cpu_count()
    gscv = GridSearchCV(estimator, params, cv=ss,
                        n_jobs=cpu_count() - 2, iid=True, refit=True,
                        verbose=1)

    if distributed == 2:
        register_parallel_backend('distributed', DistributedBackend)
        with parallel_backend('distributed',
                              scheduler_host=scheduler_host):
            gscv.fit(x)
    else:
        gscv.fit(x)

    return gscv


def _get_params_dict(estimator, dict_penalty_range):
    if estimator.dict_penalty is None:
        return {}

    return {'dict_penalty': (estimator.dict_penalty.
                                 make_grid(dict_penalty_range[0],
                                           dict_penalty_range[1],
                                           dict_penalty_range[2]))}


def _get_params_coeff(estimator, coeff_penalty_range,
                      representation_learning=0):
    if estimator.coeff_penalty is None:
        return {}

    if representation_learning:
        return {'penalty': (estimator.coeff_penalty.
                             make_grid(coeff_penalty_range[0],
                                       coeff_penalty_range[1],
                                       coeff_penalty_range[2]))}

    return {'coeff_penalty': (estimator.coeff_penalty.
                             make_grid(coeff_penalty_range[0],
                                       coeff_penalty_range[1],
                                       coeff_penalty_range[2]))}


def _get_params(estimator, dict_penalty_range, coeff_penalty_range):
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
    return {'dict_penalty': (estimator.dict_penalty.
                             make_grid(dict_penalty_range[0],
                                       dict_penalty_range[1],
                                       dict_penalty_range[2])),
            'coeff_penalty': (estimator.coeff_penalty.
                              make_grid(coeff_penalty_range[0],
                                        coeff_penalty_range[1],
                                        coeff_penalty_range[2]))}


def _check_estimator(estimator):
    if not ((isinstance(estimator, DictionaryLearning)) or
            isinstance(estimator, RepresentationLearning)):
        logging.error('Unknown estimator.')
        raise TypeError('Unknown estimator.')


def _check_range(r):
    if r is None:
        return
    if not len(r) == 3:
        logging.error('Too few elements in range specification.')
        raise ValueError('Too few elements in range specification.')

    if r[0] < 0:
        logging.error('The minimum number for the range is 0.')
        raise ValueError('The minimum number for the range is 0.')

    if r[2] > 10:
        logging.warning('Using more than ten values for each penalty may '
                        'require a long computational time.')


def _check_scheduler(s):
    a = s.split(".")
    l = a[-1].split(":")
    IP = a[0:3] + [l[0]]
    port = l[1]

    accepted = True
    if len(IP) != 4:
        accepted = False
    for x in IP:
        if not x.isdigit():
            accepted = False
        i = int(x)
        if i < 0 or i > 255:
            accepted = False
    if not accepted:
        logging.warning("The given IP does not respect the requirements."
                        "Parallel version will be executed")
        return 0
    return 2
