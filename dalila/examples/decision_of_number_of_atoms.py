from __future__ import print_function
from dalila.datasets.breast_cancer import breast_cancer_data
from dalila.datasets.casual_synthetic \
    import synthetic_data_non_negative, synthetic_data_negative
from dalila.parameters_research import eigenvalues_method, bic_method

print("\n\nExperiment with synthetic non negative dataset")
X = synthetic_data_non_negative(gaussian_noise=0.5)
k1 = eigenvalues_method(X)
k2 = bic_method(X, non_negative='both')
print("Number obtain with PCA ", k1)
print("Number obtain with BIC cross validation", k2)

print("\n\nExperiment with synthetic dataset")
X = synthetic_data_negative(gaussian_noise=0.5)
k1 = eigenvalues_method(X)
k2 = bic_method(X, non_negative='coeff')
print("Number obtain with PCA ", k1)
print("Number obtain with BIC cross validation", k2)

print("\n\nExperiment with breast cancer dataset")
X, t = breast_cancer_data()
k1 = eigenvalues_method(X)
k2 = bic_method(X, non_negative='both')
print("Number obtain with PCA ", k1)
print("Number obtain with BIC cross validation", k2)
