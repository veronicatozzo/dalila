import numpy as np

from scipy import signal
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt


def synthetic_data_non_negative(gaussian_noise=1,
                                random_state=None):
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
        coeffs = rnd.random_sample(number_of_nonzero_elements)*100
        coefficients[indices, i] = coeffs

    # create matrix
    v = np.dot(atoms, coefficients)

    # add noise
    v_tilde = v + np.random.normal(0, gaussian_noise,
                                   (number_of_features, number_of_samples))
    v_tilde[np.where(v_tilde < 0)] = 0

    return v_tilde.T


def synthetic_data_negative(gaussian_noise=1, n_features=60,
                            n_samples=100, random_state=None):
    # type: (object, object, object, object) -> object
    """
    Generates synthetic data for dictionary learning tests.

    This function generates a matrix generated from 10 basic atoms linearly
    combined with random weights sparse over the atoms. A certain level of
    gaussian noise is added to the signal.

    Parameters
    ----------
    gaussian_noise: float, optional
        The level of noise to add to the synthetic data.

    n_features: int, optional
        Number of features the output matrix must have.

    n_samples: int, optional
        Number of samples for the output matrix.

    random_state: RandomState or int, optional
        RandomState or seed used to generate RandomState for the
        reproducibility of data. If None each time RandomState is randomly
        initialised.

    Returns
    -------
    array_like, shape=(n_samples, number_of_features)
        Generated matrix of data
    """
    plt.close("all")
    number_of_atoms = 5
    rnd = check_random_state(random_state)

    atoms = np.empty([n_features, number_of_atoms])

    t = np.linspace(0, 1, n_features)
    #atoms[:, 0] = signal.sawtooth(2*np.pi*5*t)
    atoms[:, 0] = np.sin(2 * np.pi * t)
    atoms[:, 1] = np.sin(2*np.pi*t - 15)
    #atoms[:, 2] = signal.gaussian(n_features, 5)
    #atoms[:, 2] = signal.square(2 * np.pi * 5 * t)
    atoms[:, 2] = np.abs(np.sin(2 * np.pi * t))

    z = signal.gausspulse(t - 0.5, fc=5, retquad=True, retenv=True)
    #atoms[:, 3] = z[0]
    #atoms[:, 4] = z[1]
    atoms[:, 3] = z[2]
    #atoms[:, 6] = signal.ricker(n_features, 5)
    atoms[:, 4] = np.roll(np.sign(z[2] - 0.5), 10)

    # for i in range(0, number_of_atoms):
    #     plt.figure()
    #     plt.plot(atoms[:, i])
    # plt.show()

    # create sparse coefficients
    coefficients = np.zeros([number_of_atoms, n_samples])
    for i in range(0, n_samples):
        number_of_nonzero_elements = rnd.randint(2, 4)
        indices = rnd.choice(range(0, number_of_atoms),
                             number_of_nonzero_elements,
                             replace=False)
        coeffs = rnd.random_sample(number_of_nonzero_elements) * 10
        coefficients[indices, i] = coeffs

    # create matrix
    v = np.dot(atoms, coefficients)

    # add noise
    v_tilde = v + np.random.normal(0, gaussian_noise,
                                   (n_features, n_samples))
    return v_tilde.T

