from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.utils import check_random_state


def group_lasso_dataset_generator(n_samples=100, n_features=100,
                                  gaussian_noise=0.5, random_state=None):
    """
        Generates synthetic data for group lasso tests.

        This function generates a matrix generated from 7 basic atoms, grouped
        as [0, 1, 3], [2, 4, 5], linearly combined with random weights.
        A certain level of gaussian noise is added to the signal.

        Parameters
        ----------
         n_samples: int, optional
        Number of samples for the output matrix.

        n_features: int, optional
            Number of features the output matrix must have.

        gaussian_noise: float, optional
            The level of noise to add to the synthetic data.

        random_state: RandomState or int, optional
            RandomState or seed used to generate RandomState for the
            reproducibility of data. If None each time RandomState is randomly
            initialised.

        Returns
        -------
        array_like, shape=(n_samples, n_features)
            Generated matrix of data

        array_like, shape=(n_samples, 7)
            Coefficients

        array_like, shape=(7, n_features)
            Dictionary
        """
    rnd = check_random_state(random_state)
    number_of_atoms = 6

    atoms = np.empty([n_features, number_of_atoms])
    t = np.linspace(0, 1, n_features)
    atoms[:, 0] = signal.sawtooth(2 * np.pi * 5 * t)
    atoms[:, 1] = np.sin(2 * np.pi * t)
    atoms[:, 2] = np.sin(2 * np.pi * t - 15)
    atoms[:, 3] = signal.gaussian(n_features, 5)
    atoms[:, 4] = signal.square(2 * np.pi * 5 * t)
    atoms[:, 5] = np.abs(np.sin(2 * np.pi * t))

    groups = [[0, 1, 3], [2, 4, 5]]

    signals = np.empty((n_samples, n_features))
    coefficients = np.zeros((n_samples, number_of_atoms))
    for i in range(n_samples // 2):
        coeffs = rnd.random_sample(len(groups[0])) * 10
        coefficients[i, groups[0]] = coeffs

    for i in range(n_samples // 2, n_samples):
        coeffs = rnd.random_sample(len(groups[1])) * 10
        coefficients[i, groups[1]] = coeffs

    signals = coefficients.dot(atoms.T)

    return signals, coefficients, atoms.T


def sparse_signal_generator(n_samples, n_features, frequencies,
                            support_atoms, shift=True):
    # TODO: sistemare questa documentazione
    """ The following function generates signals using sawtooth and sin

    Parameters
    -------------------
    n_samples : int
        number of signals to be generated

    n_features : int
        length of the time series (number of points)

    frequencies :
        number of frequencies (to be used for the def of atoms)

    support_atoms:
        qualcosa

    shift :
        if true shifted atoms, else fixed

    Returns
    -------------------
    multichannel_matrix : np.array(n_features, n_samples)
         matrix of signals
    atoms_matrix :  np.array(n_features, number_of_atoms)
         matrix of signals
    """

    f_array = np.linspace(4. / n_features, 40. / n_features, frequencies)
    atom_shape = 2

    if shift:
        n_shifts = n_features - support_atoms
    else:
        n_shifts = 1

    n_atoms = frequencies * atom_shape * n_shifts
    _low = int(0.4 * n_atoms)
    _high = int(0.7 * n_atoms)
    selected_atoms = np.random.randint(low=_low, high=_high, size=(10,))
    atoms = np.zeros((n_features, n_atoms))
    time_vector = np.arange(support_atoms)
    diff_supp = n_features - support_atoms

    for i in range(frequencies):
        temp1 = np.sin(f_array[i] * time_vector)
        temp2 = signal.sawtooth(f_array[i] * time_vector)
        norm1 = np.linalg.norm(np.pad(temp1, (0, diff_supp), mode='constant'))
        norm2 = np.linalg.norm(np.pad(temp2, (0, diff_supp), mode='constant'))
        for j in range(n_shifts):
            atoms[:, i * n_shifts + j] = np.pad(temp1, (j, diff_supp - j),
                                                mode='constant') / norm1
            atoms[:, i * n_shifts + j + frequencies * n_shifts] = \
                np.pad(temp2, (j, diff_supp - j), mode='constant') / norm2

    multichannel_signal = np.zeros((n_features, n_samples))
    for i in range(n_samples):
        random_atoms = np.random.choice(selected_atoms, size=5)
        weight = 10 * np.random.randn(5, )
        multichannel_signal[:, i] = np.dot(atoms[:, random_atoms], weight)

    np.save('signal_gen', multichannel_signal)
    np.save('atom_gen', atoms)

    return multichannel_signal, atoms


def synthetic_data_non_negative(gaussian_noise=1, random_state=None):
    """
    Generates synthetic non-negative data for dictionary learning tests.

    This function generates a matrix generated from 7 basic atoms linearly
    combined with random weights sparse over the atoms. A certain level of
    gaussian noise is added to the signal.

    Parameters
    ----------
    gaussian_noise: float, optional
        The level of noise to add to the synthetic data.

    random_state: RandomState or int, optional
        RandomState or seed used to generate RandomState for the
        reproducibility of data. If None each time RandomState is randomly
        initialised.

    Returns
    -------
    array_like, shape=(80, 96)
        Generated matrix of data

    array_like, shape=(80, 7)
        Coefficients

    array_like, shape=(7, 96)
        Dictionary
    """

    number_of_features = 96
    number_of_samples = 80
    number_of_atoms = 7
    rnd = check_random_state(random_state)

    atoms = np.empty([number_of_features, number_of_atoms])
    atoms[:, 0] = np.transpose(
        np.concatenate((np.ones([30, 1]), np.zeros([66, 1]))))
    atoms[:, 1] = np.transpose(
        np.concatenate((np.zeros([60, 1]), np.ones([36, 1]))))
    atoms[:, 2] = np.transpose(np.concatenate(
        (np.zeros([24, 1]), np.ones([30, 1]), np.zeros([42, 1]))))
    atoms[:, 3] = signal.gaussian(96, 5)
    atoms[:, 4] = np.transpose(np.concatenate((np.zeros([17, 1]),
                                               np.ones([15, 1]),
                                               np.zeros([30, 1]),
                                               np.ones([24, 1]),
                                               np.zeros([10, 1]))))
    atoms[:, 5] = np.roll(signal.gaussian(96, 5), 30)
    atoms[:, 6] = signal.gaussian(96, 8)
    atoms[0:50, 6] = 0

    sums = np.sum(atoms, axis=0)
    atoms = atoms / sums

    # create sparse coefficients
    coefficients = np.zeros([number_of_atoms, number_of_samples])
    for i in range(0, number_of_samples):
        number_of_nonzero_elements = rnd.randint(2, 4)
        indices = rnd.choice(range(0, 7), number_of_nonzero_elements,
                             replace=False)
        coeffs = rnd.random_sample(number_of_nonzero_elements) * 100
        coefficients[indices, i] = coeffs

    # create matrix
    v = np.dot(atoms, coefficients)

    # add noise
    v_tilde = v + np.random.normal(0, gaussian_noise,
                                   (number_of_features, number_of_samples))
    v_tilde[np.where(v_tilde < 0)] = 0

    return v_tilde.T, coefficients.T, atoms.T


def synthetic_data_negative(n_samples=100, n_features=60,
                            gaussian_noise=1, random_state=None):
    """
    Generates synthetic data for dictionary learning tests.

    This function generates a matrix generated from 10 basic atoms linearly
    combined with random weights sparse over the atoms. A certain level of
    gaussian noise is added to the signal.

    Parameters
    ----------
    n_samples: int, optional
        Number of samples for the output matrix.

    n_features: int, optional
        Number of features the output matrix must have.

    gaussian_noise: float, optional
        The level of noise to add to the synthetic data.

    random_state: RandomState or int, optional
        RandomState or seed used to generate RandomState for the
        reproducibility of data. If None each time RandomState is randomly
        initialised.

    Returns
    -------
    array_like, shape=(n_samples, number_of_features)
        Generated matrix of data

    array_like, shape=(n_samples, 5)
        Coefficients

    array_like, shape=(5, n_features)
        Dictionary
    """

    plt.close("all")
    n_atoms = 5
    rnd = check_random_state(random_state)

    atoms = np.empty([n_features, n_atoms])

    t = np.linspace(0, 1, n_features)
    atoms[:, 0] = np.sin(2 * np.pi * t)
    atoms[:, 1] = np.sin(2 * np.pi * t - 15)
    atoms[:, 2] = np.abs(np.sin(2 * np.pi * t))
    z = signal.gausspulse(t - 0.5, fc=5, retquad=True, retenv=True)
    atoms[:, 3] = z[2]
    atoms[:, 4] = np.roll(np.sign(z[2] - 0.5), 10)

    # create sparse coefficients
    coefficients = np.zeros([n_atoms, n_samples])
    for i in range(0, n_samples):
        number_of_nonzero_elements = rnd.randint(2, 4)
        indices = rnd.choice(range(0, n_atoms),
                             number_of_nonzero_elements,
                             replace=False)
        coeffs = rnd.random_sample(number_of_nonzero_elements) * 10
        coefficients[indices, i] = coeffs

    # create matrix
    v = np.dot(atoms, coefficients)

    # add noise
    v_tilde = v + np.random.normal(0, gaussian_noise,
                                   (n_features, n_samples))
    return v_tilde.T, coefficients.T, atoms.T
