import numpy as np

def rank_basis(X_basis):

    dim = X_basis[0].shape[0]
    X_flatten = [X.reshape(dim*dim) for X in X_basis]
    X = np.stack(X_flatten)

    rank = np.linalg.matrix_rank(X)
    return rank
