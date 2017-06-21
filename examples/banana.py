from __future__ import division
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from dalila.dictionary_learning import DictionaryLearning
from dalila.penalty import GroupLassoPenalty
import logging
from dalila.utils import compute_correlation

logging.basicConfig(level=logging.INFO)

n_samples = 100
n_features = 100
number_of_atoms = 6

atoms = np.empty([n_features, number_of_atoms])
t = np.linspace(0, 1, n_features)
atoms[:, 0] = signal.sawtooth(2*np.pi*5*t)
atoms[:, 1] = np.sin(2 * np.pi * t)
atoms[:, 2] = np.sin(2 * np.pi * t - 15)
atoms[:, 3] = signal.gaussian(n_features, 5)
atoms[:, 4] = signal.square(2 * np.pi * 5 * t)
atoms[:, 5] = np.abs(np.sin(2 * np.pi * t))


groups = [[0, 1, 3],  [2, 4, 5]]

decompositions = []
for i in range(14):
    decompositions.append(np.load("decomposition"+str(i)+".npy"))


for d in decompositions:
    C = d[0]
    D = d[1]
    new_D = np.empty_like(D)
    new_C = np.empty_like(C)
    found = []
    for i in range(6):
        corrs = compute_correlation(atoms[:,i].T, D, found)
        index = np.argmax(corrs)
        new_D[i, :] = D[index,:]
        new_C[:,i] = C[:, index]

        found.append(index)

    plt.pcolor(new_C, cmap=plt.cm.Reds)
    plt.show()
