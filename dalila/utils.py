import numpy as np
import sys
import logging
from scipy import signal


def non_negative_projection(x, nn, matrix_type=None):
    if matrix_type is None and nn:
        np.maximum(x, 0, out=x)
    if nn == 'both' or nn == matrix_type:
        np.maximum(x, 0, out=x)
    return x


def check_non_negativity(nn, x):
    if nn == 'both':
        if np.min(x) < 0:
            logging.error('The matrix of signals to decompose has '
                          'negative numbers, impossible to use non-negative'
                          ' matrix factorization.')
            sys.exit(0)
    if not (nn == 'both' or nn == 'none' or nn == 'coeff'):
        logging.error("Unknown option for non_negativy, please use"
                      "one of {'none', 'both', 'coeff'}.")
        sys.exit(0)


def compute_clusters_and_silhouettes(Ds, Cs):
    all_dictionaries, all_coefficients, centroids, clusters = \
        _clustering(Ds, Cs)
    silhouettes = _silhouette(clusters)

    sums = np.zeros_like(all_coefficients[0])

    for i in range(0, len(all_coefficients)):
        sums = sums + all_coefficients[i]
    mean_coeffs = sums / len(all_coefficients)

    return all_dictionaries, all_coefficients, centroids, \
           mean_coeffs, np.mean(silhouettes)


def _clustering(Ds, Cs):
    # take the last element (not clustered yet)
    iterations = len(Ds)
    index_last = iterations - 1
    D = Ds[index_last]
    C = Cs[index_last]

    # if first iteration the dictionary is already the cluster
    if iterations == 1:
        return Ds, Cs, D, D

    # take dimensions
    number_of_atoms, number_of_features = D.shape

    # recompute centroids and clusters
    clusters = []
    sums = np.zeros((number_of_atoms, number_of_features))
    for k in range(number_of_atoms):
        cluster = []
        for d in range(iterations - 1):
            sums[k, :] = sums[k, :] + Ds[d][k, :]
            cluster.append(Ds[d][k, :])
        clusters.append(cluster)

    centroids = sums / (iterations - 1)

    # find the nearest element in D to each centroids
    new_order_for_atoms = np.zeros(number_of_atoms)
    used_atoms = []
    for c in range(number_of_atoms):
        distances = compute_correlation(centroids[c, :], D, used_atoms)
        index = np.argmax(distances)
        new_order_for_atoms[c] = index
        centroids[c, :] = (sums[c, :] + D[index, :]) / iterations
        clusters[c].append(D[index, ])
        used_atoms.append(index)

    # order D_last and H_last w.r.t. ordered_D
    new_D = np.zeros_like(D)
    new_C = np.zeros_like(C)
    for i in range(number_of_atoms):
        new_D[i, :] = D[int(new_order_for_atoms[i]), :]
        new_C[:, i] = C[:, int(new_order_for_atoms[i])]

    Ds[index_last] = new_D
    Cs[index_last] = new_C

    return Ds, Cs, centroids, clusters


def _silhouette(clusters):
    silhouettes = np.zeros(len(clusters))
    for c in range(len(clusters)):
        C = clusters[c]
        others = [x for i, x in enumerate(clusters) if i != c]
        silhouettes[c] = single_cluster_silhouette(C, others)
    return silhouettes


def single_cluster_silhouette(A, others):
    single_point_silhouette = np.zeros(len(A))
    for i in range(len(A)):
        a_i = _average_distance(A[i], A)

        d_i = np.zeros(len(others))
        for c in range(len(others)):
            d_i[c] = _average_distance(A[i], others[c])
        b_i = np.max(d_i)

        if a_i > b_i:
            single_point_silhouette[i] = 1 - b_i / a_i
        elif a_i == b_i:
            single_point_silhouette[i] = 0
        else:
            single_point_silhouette[i] = a_i / b_i - 1
    return np.mean(single_point_silhouette)


def _average_distance(point, cluster):
    sum_of_distances = 0
    for new_point in cluster:
        sum_of_distances = sum_of_distances + np.dot(point, new_point)
    return sum_of_distances / len(cluster)


def compute_correlation(x, set_to_compare, not_to_consider):
    """
    Computes correlations between x and the set of other signals.

    Given a signal x it computes the correlations with all the other signals
    in set_to_compare. The correlation of the signals whose indices are in
    not_to_consider will be -inf.

    Parameters
    ----------
    x: array-like, shape=(1, n_features)
        Signal to compare.

    set_to_compare: array-like or list, shape=(n_samples, n_features)
        Matrix of signals on which compute correlation.

    not_to_consider: list of int
        Indices of the matrix set_to_compare that do not have to be
        considered. The correlation for this signals will be set to -inf.

    Returns
    -------
    array-like, shape=(n_samples)
        The correlations of all the signals in set_to_compare with x.
    """

    set_to_compare = np.array(set_to_compare)
    correlations = np.zeros(set_to_compare.shape[0])
    for a in range(set_to_compare.shape[0]):
        if a in not_to_consider:
            correlations[a] = -float("inf")
        else:
            vector = set_to_compare[a, :]
            correlations[a] = (np.dot(x, vector.T))
    return correlations


def sparse_signal_generator(n, length, frequencies, support_atoms, shift=True):

    """ The following function generates signals using sawtooth and sin
    Parameters
    -------------------
    number : number of signals to be generated
    length : length of the time series (number of points)
    frequencies : number of frequencies (to be used for the def of atoms)
    shift = if true shifted atoms, else fixed

    Returns
    -------------------
    multichannel_matrix : matrix of signals np.array(length, n)
    atoms_matrix : dictionary np.array(length, number_of_atoms)
    """

    f_array = np.linspace(4./length, 40./length, frequencies)
    atom_shape = 2

    if shift:
        n_shifts = length - support_atoms
    else:
        n_shifts = 1

    number_of_atoms = frequencies * atom_shape * n_shifts
    _low = int(0.4 * number_of_atoms)
    _high = int(0.7 * number_of_atoms)
    selected_atoms = np.random.randint(low=_low, high=_high, size=(10,))
    atoms = np.zeros((length, number_of_atoms))
    time_vector = np.arange(support_atoms)
    diff_supp = length - support_atoms

    for i in range(frequencies):
        temp1 = np.sin(f_array[i] * time_vector)
        temp2 = signal.sawtooth(f_array[i] * time_vector)
        norm1 = np.linalg.norm(np.pad(temp1, (0, diff_supp), mode='constant'))
        norm2 = np.linalg.norm(np.pad(temp2, (0, diff_supp), mode='constant'))
        for j in range(n_shifts):
            atoms[:, i*n_shifts+j] = np.pad(temp1, (j, diff_supp-j), mode='constant')/norm1
            atoms[:, i*n_shifts+j+frequencies*n_shifts] = np.pad(temp2, (j, diff_supp-j), mode='constant')/norm2

    multichannel_signal = np.zeros((length, n))
    for i in range(n):
        random_atoms = np.random.choice(selected_atoms, size=5)
        weight = 10 * np.random.randn(5,)
        multichannel_signal[:, i] = np.dot(atoms[:, random_atoms], weight)

    np.save('signal_gen', multichannel_signal)
    np.save('atom_gen', atoms)

    return multichannel_signal, atoms
