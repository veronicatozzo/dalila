from __future__ import division

import logging
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from dalila.dictionary_learning import DictionaryLearning
from dalila.penalty import GroupLassoPenalty
from dalila.dataset_generator import group_lasso_dataset_generator



logging.basicConfig(level=logging.INFO)

signals, coefficients, atoms = group_lasso_dataset_generator()
plt.pcolor(coefficients, cmap=plt.cm.Reds)
plt.show()


decompositions = []
for i in np.linspace(0.1, 10, 15):
    estimator = DictionaryLearning(k=6,
                                   dict_penalty=None,
                                   coeff_penalty=GroupLassoPenalty(groups, i),
                                   non_negativity="coeff")

    estimator.fit(signals, n_iter=20000)
    decompositions.append(estimator.decomposition())
    print("finito", i)
    print("reconstruction error", estimator.reconstruction_error())
    print("\n\n")


for (i, d) in enumerate(decompositions):
    # it is necessary to reorder the atoms as they are declared in atoms to
    # see if the algorithm works
    C = d[0]
    D = d[1]
    new_D = np.empty_like(D)
    new_C = np.empty_like(C)
    found = []
    for i in range(6):
        corrs = compute_correlation(atoms[:, i].T, D, found)
        index = np.argmax(corrs)
        new_D[i, :] = D[index, :]
        new_C[:, i] = C[:, index]

        found.append(index)

    plt.pcolor(new_C, cmap=plt.cm.Reds)
    plt.show()
