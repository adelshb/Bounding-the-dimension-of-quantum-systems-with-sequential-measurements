import numpy as np
from itertools import combinations, product
import scipy
import random
from scipy.stats import unitary_group

import warnings
warnings.filterwarnings('ignore')

def generate_basis(dim,
                num_obs,
                len_seq,
                out_max = 1,
                batch_size=100,
                seq_method="all_sequences",
                sel_sequences = [2],
                remove_last_out = True,
                compute_rank = True):

    if out_max + 1 > dim:
        print("Dimension is not high enough.")
        return

    scipy.random.seed()
    X_basis = [rand_moment(dim, num_obs, len_seq, out_max, seq_method=seq_method, sel_sequences=sel_sequences, remove_last_out=remove_last_out) for __ in range(batch_size)]
    if compute_rank == True:
        rank = rank_basis(X_basis)
        return X_basis, rank
    elif compute_rank == False:
        return X_basis

def rand_moment(dim=2,
             num_obs=3,
             len_seq=2,
             out_max=1,
             seq_method="all_sequences",
             sel_sequences = [2],
             remove_last_out = True):

    if seq_method == "all_sequences":
        sequences = all_seq(num_obs, r_max=len_seq, out_max=out_max, remove_last_out=remove_last_out)
    elif seq_method == "sep_seq":
        sequences = sep_seq(num_m=num_obs, num_p=len_seq, out_max=out_max)
    elif seq_method == "sel_sequences":
        sequences = []
        for r in list(set(sel_sequences)):
            sequences.append(sel_seq(num_obs, r, out_max=out_max, remove_last_out=remove_last_out))
        sequences = sum(sequences, [])

    rho = rand_rho(dim)
    P = [rand_projs(dim, out_max) for __ in range(num_obs)]

    # Chi representation of the moment matrix
    if remove_last_out:
        X = np.eye(len(sequences)+1, dtype=complex)
        for i, seq_row in enumerate(sequences):
            Pi = proj_mul([P[k] for k in seq_row[1]], seq_row[0])
            X[0,i+1] = np.trace(Pi @ np.eye(dim) @ rho)
            X[i+1,0] = np.trace(np.eye(dim) @ np.conjugate(Pi.T) @ rho)
            for j, seq_col in enumerate(sequences):
                Pj= proj_mul([P[k] for k in seq_col[1]], seq_col[0])
                X[i+1,j+1] = np.trace(Pi @ np.conjugate(Pj.T) @ rho)
        return X
    # M representation of the moment matrix
    else:
        X = np.zeros((len(sequences),len(sequences)), dtype=complex)
        for i, seq_row in enumerate(sequences):
            Pi = proj_mul([P[k] for k in seq_row[1]], seq_row[0])
            for j, seq_col in enumerate(sequences):
                Pj= proj_mul([P[k] for k in seq_col[1]], seq_col[0])
                X[i,j] = np.trace(Pi @ np.conjugate(Pj.T) @ rho)
        return X
        # X = np.eye(len(sequences)+1, dtype=complex)
        # for i, seq_row in enumerate(sequences):
        #     Pi = proj_mul([P[k] for k in seq_row[1]], seq_row[0])
        #     X[0,i+1] = np.trace(Pi @ np.eye(dim) @ rho)
        #     X[i+1,0] = np.trace(np.eye(dim) @ np.conjugate(Pi.T) @ rho)
        #     for j, seq_col in enumerate(sequences):
        #         Pj= proj_mul([P[k] for k in seq_col[1]], seq_col[0])
        #         X[i+1,j+1] = np.trace(Pi @ np.conjugate(Pj.T) @ rho)
        # return X

def rand_projs(dim, out_max):
    if dim == 1:
        return 1
    elif dim==2:
        D = np.array([[1, 0],[0, 0]])
        U = unitary_group.rvs(dim)
        return [U @ D @ np.conjugate(U.T), U @ (np.eye(dim)-D) @ np.conjugate(U.T)]
    elif dim>2:
        projs = []
        eigenvals = [0]*(out_max + 1)
        while 0 in eigenvals:
            eigenvals = list(random_ints_with_sum(dim, out_max + 1))
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

def rank_basis(X_basis):
    dim = X_basis[0].shape[0]
    X_flatten = [X.flatten() for X in X_basis]
    X = np.stack(X_flatten)
    rank = np.linalg.matrix_rank(X)
    return rank

def all_seq(n,r_max=2,
            out_max=1,
            remove_last_out=True):

    if remove_last_out:
        out_max += -1
    arr = [i for i in range(n)]
    seq = []
    for r in range(1,r_max+1):
        settings = list(product(arr, repeat=r))
        outcomes = list(product([i for i in range(out_max+1)], repeat=r))
        seq.append([(r,s) for r in outcomes for s in settings])
    seq = sum(seq,[])
    return seq

def sel_seq(num_obs,
            len_seq=2,
            out_max=1,
            remove_last_out=True):

    if remove_last_out:
        out_max += -1
    arr = [i for i in range(num_obs)]
    seq = []

    settings = list(product(arr, repeat=len_seq))
    outcomes = list(product([i for i in range(out_max+1)], repeat=len_seq))
    seq.append([(res,set) for res in outcomes for set in settings])
    seq = sum(seq,[])
    return seq

def sep_seq(num_m=2,
            num_p=3,
            out_max=1):

    settings = [[i for i in range(j,j+num_m)] for j in range(0,num_p*num_m,num_m)]
    seq = []
    for p in range(1,num_p+1):
        pos_settings = list(product([i for i in range(num_m)], repeat=p))
        comb_parties = list(combinations(range(1,num_p+1), p))
        for comb_p in comb_parties:
            for pos in pos_settings:
                temp_set = [settings[comb_p[ind]-1][pos[ind]] for ind in range(len(pos))]
                outcomes = list(product([i for i in range(out_max+1)], repeat=p))
                seq.append([(o, tuple(temp_set)) for o in outcomes])
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
