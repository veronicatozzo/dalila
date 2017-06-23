import numpy as np
from scipy import signal


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
