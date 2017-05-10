DALILA (DictionAry LearnIng LibrAry)
====================================

**DALILA** is a Dictionary Learning Library whose purpose is to find a decomposition of an input matrix **X** into two other matrices **D** and **C** which are respectively the *dictionary* which contains the basic atoms and the *coefficients* that are weights for the atoms. The linear combination of atoms weighted with the coefficients give an approximation of the original signal.

We propose a generic optimization algorithm that can optimize the functional with different penalties both on the dictionary and on the coefficients. The algorithm works for unsupervised and supervised dictionary learning.


Unsupervised dictionary learning
---------------------------------
Optimization of a dictionary learning problem in the following form

            ||X - DC|| + c(C) + d(D)

    
the penalties on **C** and **D** can be of different types an precisely we offer the possibility to use:

- L1-norm    
- L2-norm
- elastic-net
- Total Variation
- Group Lasso
- Non-negativity constraint

or a combination of the previous on both the dictionary and the coefficients. 


Supervised dictionary learning
------------------------------
Optimization of a discriminative dictionary learning problem in the following form


        ||X - DC|| + ||y - wC||+ c(C) + d(D) + g(w)


where **y** is a vector of classes or regression values,  **w** are the regression coefficients based on the new representation of the samples given by **C** and g(**w**) is the penalty function on the coefficients and it can be a combination of:

- L1-norm
- L2-norm
- elastic-net


Cross-validation
----------------
The library contains a procedure to analyse, on both problems, which is the best number of atoms to decompose the signal matrix.
In an unsupervised case cross-validation is performed on BIC (Bayesian Information Criterion) value computed on the reconstruction error
while in the supervised case cross-validation is performed on the classification/regression error obtained by predicting the **y** of a test set
after the computation of the coefficients with the dictionary fixed.