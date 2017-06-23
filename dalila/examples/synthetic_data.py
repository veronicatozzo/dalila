from __future__ import print_function, division
import logging
import matplotlib.pyplot as plt
from casual_synthetic import synthetic_data_non_negative, synthetic_data_negative
from dalila.dictionary_learning import DictionaryLearning,\
    StabilityDictionaryLearning
from dalila.penalty import *
from dalila.plot import plot_dictionary_atoms
from dalila.sparse_coding import SparseCoding


plt.close("all")

#filename='example.log',
logging.basicConfig( level=logging.INFO)
#X = synthetic_data_non_negative(gaussian_noise=0.5)
X = synthetic_data_non_negative(gaussian_noise=0.1)

dict_penalty = L2Penalty(0.01)# ElasticNetPenalty(0.01, 0.01, 0.3)  #
coeff_penalty = L1Penalty(0.01)
k = 8
estimator = StabilityDictionaryLearning(k=k,
                               dict_penalty=dict_penalty,
                               dict_normalization=0,
                               coeff_penalty=coeff_penalty,
                               non_negativity='both')

# best_est, best_parameters = \
#     tune_all_parameters(X, estimator, verbose=1,
#                         dict_penalty_range=(0.01, 0.09, 5),
#                         coeff_penalty_range=(0.01, 0.09, 5))
#
# # best_est.fit(X, adaptive=1, verbose=1)
#
# C, D = best_est.decomposition()
# print("reconstruction error",
#       best_est.reconstruction_error())
# print("dictionary penalty: ", str(best_parameters["dict_penalty"]))
# print("coefficients penalty: ", str(best_parameters["coeff_penalty"]))

# C, D = best_est.decomposition()
# print("reconstruction error",
#       best_est.reconstruction_error())
# print("dictionary penalty: ", str(best_parameters["dict_penalty"]))
# print("coefficients penalty: ", str(best_parameters["coeff_penalty"]))

estimator.fit(X, backtracking=0, n_iter=20000)
C, D = estimator.decomposition()
print(C.shape)
print(D.shape)
# print("reconstruction error before ", estimator.reconstruction_error())
# coefficients = SparseCoding(L0Penalty(4), non_negativity=1)
# C = coefficients.fit(X, D)

#for r in range(C.shape[0]):
 #   print(np.count_nonzero(C[r,:]))

# print("reconstruction error",
#       coefficients.reconstruction_error())

plot_dictionary_atoms(D)
