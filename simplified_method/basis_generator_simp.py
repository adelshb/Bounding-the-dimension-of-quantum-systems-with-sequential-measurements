import numpy as np
from itertools import combinations
from scipy.stats import unitary_group

def generate_basis(dim,
                sequences, # All sequences must be length 2.
                basis_size=10):

    X_basis = [rand_moment(dim, sequences) for __ in range(basis_size)]
    rank = rank_basis(X_basis)
    return X_basis, rank

def rand_moment(dim,
            sequences,
            all_sequences = True):

    num_obs = np.array(sequences).max()
    if all_sequences==True:
        sequences = all_seq(num_obs+1, r=2)

    rho = rand_rho(dim)

    A = []
    A.append(np.eye(dim))
    for k in range(num_obs):
        P_temp = rand_proj(dim)
        A_temp = np.eye(dim) - 2*P_temp
        A.append(A_temp)
    X = np.ones((num_obs+1,num_obs+1))
    for seq in sequences:
        #Only working when the length of all sequences is 2.
        X[seq[0], seq[1]] = np.real(np.trace(rho @ (A[seq[0]] @ A[seq[1]] + A[seq[1]] @ A[seq[0]]))/2)
        X[seq[1], seq[0]] = np.real(X[seq[0], seq[1]])
    return X

def rand_proj(dim):
    if dim==2:
        D = np.array([[1, 0],[ 0, 0]])
    else:
        D = np.random.randint(2, size=dim)*np.eye(dim)
    U = unitary_group.rvs(dim)
    return U @ D @ np.conjugate(U.T)

def rand_rho(dim):
    psi = np.array([1] + [0]*(dim-1))
    rho = np.outer(psi, psi)
    U = unitary_group.rvs(dim)
    return U @ rho @ np.conjugate(U.T)

def rank_basis(X_basis):

    dim = X_basis[0].shape[0]
    X_flatten = [X.reshape(dim*dim) for X in X_basis]
    X = np.stack(X_flatten)

    rank = np.linalg.matrix_rank(X)
    return rank

def all_seq(n,r=2):
    arr = [i for i in range(n)]
    seq = list(combinations(arr, r))
    return seq
