import numpy as np
from random import gauss
import itertools
from scipy.stats import unitary_group

def generate_basis(dim,
                sequences, # All sequences must be length 2.
                ):

    basis_size = 1000
    basis = [rand_moment(dim, sequences) for __ in range(basis_size)]
    return basis

def rand_moment(dim, sequences):

    num_obs = np.array(sequences).max()

    rho = rand_rho(dim)

    A = []
    for k in range(num_obs+1):
        P_temp = rand_proj(dim)
        A_temp = np.eye(dim) - 2*P_temp
        A.append(A_temp)

    X = np.zeros((len(sequences),len(sequences)))
    for seq in sequences:
        #Only working when the length of all sequences is 2.
        X[seq[0], seq[1]] = np.trace(rho @ (A[seq[0]] @ A[seq[1]] + A[seq[1]] @ A[seq[0]]))/2
        X[seq[1], seq[0]] = X[seq[0], seq[1]]
    return X

def rand_proj(dim):
    D = np.random.randint(2, size=dim)*np.eye(dim)
    U = unitary_group.rvs(dim)
    return U @ D @ np.conj(U.T)

def rand_rho(dim):
    psi = np.array([1] + [0]*(dim-1))
    rho = np.outer(psi, psi)
    U = unitary_group.rvs(dim)
    return U @ rho @ np.conj(U.T)
