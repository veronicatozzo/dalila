from __future__ import print_function, division

#from dalila.datasets.shift_invariant import shift_invariant_dataset
from dalila.shift_invariant_DL import ShiftInvariantDL
from dalila.plot import plot_dictionary_atoms

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

#x = shift_invariant_dataset()

_samples=3
n_features=1000
t = np.linspace(0, 1, n_features)
x = np.zeros((_samples, n_features))
x[0, :] = np.maximum(0, np.sin(2*np.pi*5*t))
x[1, :] = np.maximum(0, signal.square(2 * np.pi * 5* t))
x[2,:]  = np.maximum(0, signal.sawtooth(2*np.pi*5*t) )
x = x + np.random.randn(3,1000)*0.1

plt.plot(t, x[0,:])
plt.plot(t, x[1,:])
plt.plot(t, x[2,:])
plt.show()
#plot_dictionary_atoms(x[0:10, ])

estimator = ShiftInvariantDL(3, 100, True)
estimator.fit(x)
#g = estimator.get_basic_atoms()

#print(g.shape)

#plot_dictionary_atoms(g)


_samples=3
n_features=1000
t = np.linspace(0, 1, n_features)
x = np.zeros((_samples, n_features))
x[0, :] = np.maximum(0, np.sin(2*np.pi*5*t))
x[1, :] = np.maximum(0, signal.square(2 * np.pi * 5* t))
x[2,:]  = np.maximum(0, signal.sawtooth(2*np.pi*5*t) )

final_data = np.zeros((100, n_features))
for n in range(100):
    indices = np.random.randint(0, 3, 2)
    coefficients = abs(np.random.randn(2, 1))
    final_data[n, :] = np.sum(coefficients*x[indices, ],
                              axis=0)
final_data += np.random.randn(100, 1000)*0.1
#plot_dictionary_atoms(x[0:10, ])

estimator = ShiftInvariantDL(3, 100, False)
estimator.fit(x)
#g = estimator.get_basic_atoms()

#print(g.shape)

#plot_dictionary_atoms(g)