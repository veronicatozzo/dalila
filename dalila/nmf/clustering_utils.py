from __future__ import division

import operator

import numpy as np
from itertools import combinations
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test
from lifelines.utils import survival_table_from_events
from lifelines import KaplanMeierFitter

from sklearn.cluster import KMeans, SpectralClustering


def jaccard_index(A, B):
    A = set(A)
    B = set(B)
    intr = A.intersection(B)
    union = A.union(B)
    return len(list(intr))/len(list(union))


def jaccard_matrix(clusters1, clusters2):
    groups1 = []
    for i in np.unique(clusters1):
        groups1.append(np.where(clusters1==i)[0])
    groups2 = []
    for i in np.unique(clusters2):
        groups2.append(np.where(clusters2==i)[0])

    res = np.zeros((len(groups1), len(groups2)))
    for i in range(len(groups1)):
        for j in range(len(groups2)):
            res[i,j] = jaccard_index(groups1[i], groups2[j])
    return res


def aggragate_clusters_on_pvalue(clusters, survival, status):
    new_clusters = clusters.copy()
    p_values = []
    many = []
    layers = []
    while len(np.unique(new_clusters))>1:
        numbers_at_risk.append([])
        res_pair = pairwise_logrank_test(survival, new_clusters, status)
        pairs = {}
        for i in range(res_pair.shape[0]):
            for j in range(i+1, res_pair.shape[0]):
                pairs[res_pair.index[i],res_pair.columns[j]] = res_pair.iloc[i,j].p_value
        sorted_x = sorted(pairs.items(), key=operator.itemgetter(1))
        e = sorted_x[-1]
        aux = new_clusters.copy()
        aux[np.where(new_clusters==e[0][1])[0]] = e[0][0]
        layers.append(aux)
        new_clusters = aux.copy()
        res = multivariate_logrank_test(survival, new_clusters, status)
        many.append(len(np.unique(new_clusters)))
        p_values.append(res.p_value)

    return zip(layers, many, p_values)

def spectral_clustering(G):
    SP = SpectralClustering(affinity='precomputed')
    SP.fit(self.adjacency)
    self.labels_ = SP.labels_
    return self.labels_


def get_clusters(G, mode='hard'):
    """
    Params
    ------
    G: array_like, shape=(n, k)
        The factorization matrix for n samples and k factors.
    mode: string, optional, default='hard'
        Modalitiy to perform clustering, default is hard clustering. Options:
        - 'hard': hard clustering
        - 'kmeans': kmeans clustering
        - 'normalization': the columns are normalized (subtraction of mean and
           rescaled of sd) and then hard clustering is performed
    """
    Gc = G.copy()
    if mode.lower() == 'hard':
        return np.argmax(G, axis=1)
    if mode.lower() == 'kmeans':
        kmeans = KMeans(n_clusters=G.shape[1]).fit(G)
        return kmeans.labels_
    if mode.lower() == 'normalization':
        Gc -= np.mean(Gc, axis=0)
        Gc /= np.std(Gc, axis=0)
        return np.argmax(Gc, axis=1)
    #if mode.lower() == 'sum':
#        Gc -= np.mean(Gc)


def erdos_renyi(n, m):
    """
    n: int,
        Number of nodes
    m: int,
        Number of edges
    """
    comb = np.array(list(combinations(np.arange(0, n), 2)))
    np.random.shuffle(comb)
    m = int(m)
    selected_comb = comb[:m]
    x = [c[0] for c in selected_comb]
    y = [c[1] for c in selected_comb]
    network = np.zeros((n,n))
    network[x, y] = 1
    network[y, x] = 1

    return network

def dispersion_coefficient_rho(X):
    coeff = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            coeff += 4*(X[i,j] - 0.5)**2
    return coeff/X.shape[0]**2


def dispersion_coefficients_eta_v(X, k):
    n = X.shape[0]
    non_diag = (np.ones(shape=X.shape) - np.identity(X.shape[0])).astype(bool)
    ravelled = X[np.where(non_diag)]
    eta = np.var(ravelled)
    eta /= ((n/k -1)/(n-1) - ((n/k -1)/(n-1))**2)

    aux = (X - 1/k)**2
    #print(aux)
    aux = aux - np.diag(np.diag(aux))
    v = np.sum(aux)
    v /= n*(n-1)*(1/k - 1/k**2)
    return eta, v




def connectivity_matrix(X):
    indices = np.argmax(X, axis=1)
    C = np.zeros_like(X)
    for r, i in enumerate(indices):
        C[r,i] = 1
    return C.dot(C.T)
