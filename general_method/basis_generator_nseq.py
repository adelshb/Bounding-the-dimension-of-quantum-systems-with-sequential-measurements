import numpy as np
from itertools import combinations, product
from scipy.stats import unitary_group

import warnings
warnings.filterwarnings('ignore')

def generate_basis(dim,
                num_obs,
                len_seq,
                out_max = 1,
                basis_size=100,
                seq_method="all_sequences"):

    X_basis = [rand_moment(dim, num_obs, len_seq, out_max, seq_method=seq_method) for __ in range(basis_size)]
    rank = rank_basis(X_basis)
    return X_basis, rank

def rand_moment(dim=2,
             num_obs=3,
             len_seq=2,
             out_max=1,
             seq_method="all_sequences"):

    if seq_method == "all_sequences":
        sequences = all_seq(num_obs, r_max=len_seq, out_max=out_max)
    elif seq_method == "sep_seq":
        sequences = sep_seq(num_m=num_obs, num_p=len_seq, out_max=out_max)
        num_obs = num_obs * len_seq

    rho = rand_rho(dim)

    P = []
    for k in range(num_obs):
        P_temp = rand_proj(dim)
        P.append(P_temp)

    X = np.eye(len(sequences)+1)
    for i, seq_row in enumerate(sequences):
        Pi = proj_mul([P[k] for k in seq_row[1]], seq_row[0])
        X[0,i+1] = np.trace(Pi @ np.eye(dim) @ rho)
        X[i+1,0] = np.trace(np.eye(dim) @ np.conjugate(Pi.T) @ rho)
        for j, seq_col in enumerate(sequences):
            Pj= proj_mul([P[k] for k in seq_col[1]], seq_col[0])
            X[i+1,j+1] = np.trace(Pi @ np.conjugate(Pj.T) @ rho)
    return X

def rand_proj(dim):
    if dim==2:
        D = np.array([[1, 0],[0, 0]])
    else:
        D = np.random.randint(2, size=dim)*np.eye(dim)
    if dim == 1:
        return 1
    else:
        U = unitary_group.rvs(dim)
        return U @ D @ np.conjugate(U.T)

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

def all_seq(n,r_max=2, out_max=1):
    arr = [i for i in range(n)]
    seq = []
    for r in range(1,r_max+1):
        settings = list(product(arr, repeat=r))
        outcomes = list(product([i for i in range(out_max+1)], repeat=r))
        seq.append([(r,s) for r in outcomes for s in settings])
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

def proj_mul(listP, outcome):
    Proj = np.eye(listP[0].shape[0])
    for i, P in enumerate(listP):
        if outcome[i] == 1:
            Proj = Proj @ P
        elif outcome[i] == 0:
            Proj = Proj @ (np.eye(listP[0].shape[0]) - P)
    return Proj
