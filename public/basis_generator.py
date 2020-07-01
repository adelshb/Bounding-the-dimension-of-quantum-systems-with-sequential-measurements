import numpy as np
from itertools import combinations, product
import scipy
import random
from scipy.stats import unitary_group

import warnings
warnings.filterwarnings('ignore')

def basis_gs(dim = 2,
            num_obs = 3,
            len_seq = 2,
            num_out = 2,
            prec = 1e-15,
            stop = 100000):

    X_basis = []
    X = rand_moment(dim,
                    num_obs,
                    len_seq,
                    num_out,
                    [len_seq],
                    True)

    X = X/LA.norm(X)
    X_basis.append(X)

    while True:

        X = rand_moment(dim,
                    num_obs,
                    len_seq,
                    num_out,
                    [len_seq],
                    True)

        for k in range(len(X_basis)):
            X -= X_basis[k]*np.sum(X_basis[k]*np.conjugate(X))

        if LA.norm(X) < prec:
            print("Nul matrix found")
            print("Number of LI moment matrices: ", len(X_basis))
            return X_basis
        else:
            X = X/LA.norm(X)
            X_basis.append(X)

        if count > args.stop:
            print("Cannot find the basis")
            return

def rand_moment(dim=2,
             num_obs=3,
             len_seq=2,
             num_out=2,
             sel_sequences = [2],
             remove_last_out = True):

    sequences = []
    for r in list(set(sel_sequences)):
        sequences.append(sel_seq(num_obs, r, num_out=num_out, remove_last_out=remove_last_out))
    sequences = sum(sequences, [])

    rho = rand_rho(dim)
    P = [rand_projs(dim, num_out) for __ in range(num_obs)]

    if remove_last_out:
        X = np.eye(len(sequences)+1, dtype=complex)
        for i, seq_row in enumerate(sequences):
            Pi = proj_mul([P[k] for k in seq_row[1]], seq_row[0])
            X[0,i+1] = np.trace(Pi @ np.eye(dim) @ rho)
            X[i+1,0] = np.conjugate(X[0,i+1])#np.trace(np.eye(dim) @ np.conjugate(Pi.T) @ rho)
            for j, seq_col in enumerate(sequences):
                Pj= proj_mul([P[k] for k in seq_col[1]], seq_col[0])
                X[i+1,j+1] = np.trace(Pi @ np.conjugate(Pj.T) @ rho)
        return X
    else:
        X = np.zeros((len(sequences),len(sequences)), dtype=complex)
        for i, seq_row in enumerate(sequences):
            Pi = proj_mul([P[k] for k in seq_row[1]], seq_row[0])
            for j, seq_col in enumerate(sequences):
                Pj= proj_mul([P[k] for k in seq_col[1]], seq_col[0])
                X[i,j] = np.trace(Pi @ np.conjugate(Pj.T) @ rho)
        return X

def rand_projs(dim, num_out):
    if dim == 1:
        return 1
    elif dim==2:
        D = np.array([[1, 0],[0, 0]])
        U = unitary_group.rvs(dim)
        return [U @ D @ np.conjugate(U.T), U @ (np.eye(dim)-D) @ np.conjugate(U.T)]
    elif dim>2:
        projs = []
        eigenvals = [0]*(num_out + 1)
        while 0 in eigenvals:
            eigenvals = list(random_ints_with_sum(dim, num_out))
        for j,val in enumerate(eigenvals):
            vec_temp = [0]*sum(eigenvals[:j]) + [1]*val + [0]*sum(eigenvals[j+1:])
            projs.append(vec_temp*np.eye(dim))

        U = unitary_group.rvs(dim)
        return [U @ D @ np.conjugate(U.T) for D in projs]

def rand_rho(dim):
    psi = np.array([1] + [0]*(dim-1))
    rho = np.outer(psi, psi)
    if dim == 1:
        return 1
    else:
        U = unitary_group.rvs(dim)
        return U @ rho @ np.conjugate(U.T)

def sel_seq(num_obs,
            len_seq=2,
            num_out=2,
            remove_last_out=True):

    if remove_last_out:
        num_out += -1
    arr = [i for i in range(num_obs)]
    seq = []

    settings = list(product(arr, repeat=len_seq))
    outcomes = list(product([i for i in range(num_out)], repeat=len_seq))
    seq.append([(res,set) for res in outcomes for set in settings])
    seq = sum(seq,[])
    return seq

def proj_mul(listP, outcomes):
    Proj = np.eye(listP[0][0].shape[0])
    for i, P in enumerate(listP):
        Proj = Proj @ P[outcomes[i]]
    return Proj

def random_ints_with_sum(dim, num):
    count=0
    while dim > 0:
        if count == num-1:
            yield dim
            break
        else:
            r = random.randint(0, dim - (num - count) + 1)
            yield r
            count += 1
            dim -= r
