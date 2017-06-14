from __future__ import division
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from dalila.dictionary_learning import DictionaryLearning
from dalila.penalty import GroupLassoPenalty
import logging


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

signals = np.empty((n_samples, n_features))
coefficients = np.zeros((n_samples, number_of_atoms))
for i in range(n_samples//2):
    coeffs = np.random.random_sample(len(groups[0])) * 10
    coefficients[i, groups[0]] = coeffs

for i in range(n_samples//2, n_samples):
    coeffs = np.random.random_sample(len(groups[1])) * 10
    coefficients[i, groups[1]] = coeffs

signals = coefficients.dot(atoms.T)

plt.pcolor(coefficients, cmap=plt.cm.Reds)
plt.show()

signals = coefficients.dot(atoms.T)

plt.pcolor(coefficients, cmap=plt.cm.Reds)
plt.show()

plt.plot(signals[0])
plt.plot(signals[1])
plt.plot(signals[2])
plt.show()

decompositions = []
for i in np.linspace(0.00001, 0.001, 15):
    estimator = DictionaryLearning(k=6,
                                   dict_penalty=None,
                                   coeff_penalty=GroupLassoPenalty(groups, i))

    estimator.fit(signals, n_iter=20000)
    decompositions.append(estimator.decomposition())

for d in decompositions:
    plt.pcolor(d[0], cmap=plt.cm.Reds)
    plt.show()
