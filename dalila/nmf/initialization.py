import numpy as np

from scipy.linalg import eig

def _init_svd(X_sum, k):
    """C. Boutsidis and E. Gallopoulos, SVD-based initialization: A head
    start for nonnegative matrix factorization, Pattern Recognition,
    Elsevier"""
    w, v = eig(X_sum)
    pos_w = np.abs(w)
    indices = np.argsort(w)[::-1]
    sorted_eigs = pos_w[indices]
    trace = np.sum(pos_w)
    tr_i = 0
    k_new = 1
    for i in range(len(w)):
        tr_i += sorted_eigs[i]
        if tr_i/trace > 0.9:
            k_new = i+1
            break

    k = min(k, k_new)

    G = []
    for i in range(k):
        xx = v[:, indices[i]]*pos_w[indices[i]]
        xp = np.maximum(xx, 0)
        print(xp)
        xn = xp - xx
        if np.linalg.norm(xp) > np.linalg.norm(xn):
            G.append(xp)
        else:
            G.append(xn)
    G = np.array(G)
    G[np.where(G<1e-10)] = 1e-10
    return G.T
