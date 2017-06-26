
import matplotlib.pyplot as plt
import numpy as np


def plot_dictionary_atoms(dictionary):
    """
    It plots the atoms composing the dictionary.

    Parameters
    ----------
    dictionary: array-like, shape=(n_atoms, n_features)

    """

    for r in range(0, dictionary.shape[0]):
        plt.figure()
        plt.plot(dictionary[r, :])
    plt.show()


def plot_atoms_as_histograms(dictionary):
    """
    It plots the atoms composing the dictionary as histograms.

    Parameters
    ----------
    dictionary: array_like, shape=(n_atoms, n_features)
    """
    for i in range(0, dictionary.shape[0]):
        fig = plt.figure()
        fig.canvas.set_window_title(str(i+1) + " atom")
        length = len(dictionary[i, :])
        x = np.asarray(range(0, length))
        w = dictionary[i, :]
        plt.hist(x, bins=length, weights=w)
        plt.xlim((0, dictionary.shape[1]))
    plt.show()

