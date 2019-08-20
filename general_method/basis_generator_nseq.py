import numpy as np
from itertools import combinations, product
from scipy.stats import unitary_group

def generate_basis(dim,
                num_obs,
                len_seq,
                out_max = 1,
                basis_size=100):

    X_basis = [rand_moment(dim, num_obs, len_seq, out_max) for __ in range(basis_size)]
    rank = rank_basis(X_basis)
    return X_basis, rank

# def rand_moment(dim,
#              num_obs,
#              len_seq):
#
#     sequences = all_seq(num_obs, r_max=len_seq)
#
#     rho = rand_rho(dim)
#
#     P = []
#     for k in range(num_obs):
#         P_temp = rand_proj(dim)
#         P.append(P_temp)
#         #P.append(np.eye(dim) - P_temp)
#
#     X = np.eye(len(sequences))
#     for i, seq_row in enumerate(sequences):
#         Pi = proj_mul([P[k] for k in seq_row])
#         for j, seq_col in enumerate(sequences[i+1:]):
#             Pj= proj_mul([P[k] for k in seq_col])
#             X[i,j] = np.trace(Pi @ np.conjugate(Pj.T) @ rho)
#             X[j,i] = X[i,j]
#     return X

def rand_moment(dim,
             num_obs,
             len_seq,
             out_max):

    sequences = all_seq(num_obs, r_max=len_seq, out_max=out_max)

    rho = rand_rho(dim)

    P = []
    for k in range(num_obs):
        P_temp = rand_proj(dim)
        P.append(P_temp)

    X = np.eye(len(sequences))
    for i, seq_row in enumerate(sequences):
        Pi = proj_mul([P[k] for k in seq_row[1]], seq_row[0])
        for j, seq_col in enumerate(sequences[i+1:]):
            Pj= proj_mul([P[k] for k in seq_col[1]], seq_col[0])
            X[i,j] = np.trace(Pi @ np.conjugate(Pj.T) @ rho)
            X[j,i] = X[i,j]
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

def proj_mul(listP, outcome):
    Proj = np.eye(listP[0].shape[0])
    for i, P in enumerate(listP):
        if outcome[i] == 1:
            Proj = Proj @ P
        elif outcome[i] == 0:
            Proj = Proj @ (np.eye(listP[0].shape[0]) - P)
    return Proj
