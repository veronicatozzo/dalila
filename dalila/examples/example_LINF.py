from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from dalila.utils import sparse_signal_generator as signal_gen
from dalila.representation_learning import RepresentationLearning
from dalila.penalty import LInfPenalty

plt.close("all")

half_n_atoms_list = np.logspace(1, 2, 5).astype(int)
reg_params_list = np.logspace(-1, 2, 10)
length = 10000
n_signals = 30

for n_atoms in half_n_atoms_list:

    signals, atoms = signal_gen(n_signals, length, n_atoms, length, False)

inf1penalty = LInfPenalty(1.)
sparse_coding = RepresentationLearning(penalty=inf1penalty)
fit_result = sparse_coding.fit(signals.T, atoms.T)

print(fit_result.score)
coef = fit_result.coefficients()

np.save("signal", signals.T)
np.save("dictionary", atoms.T)
np.save("coefficients", coef)
