import numpy as np
from basis_generator_2seq import generate_basis, all_seq

# Return the rank for a given dimension and number of measurements

def rank(basis_size=20,
            n_max = 10,
            dim_max = 10):

    rank = []
    for n in range(3, n_max):
        sequences = all_seq(n,r=2)
        rank_dim = []
        for dim in range(2, dim_max):
            __, rank_temp = generate_basis(dim, sequences, basis_size=basis_size)
            rank_dim.append(rank_temp)
        rank.append(rank_dim)
    X = np.array(rank)
    print(X)
    return X

basis_size=100
n_max = 10
dim_max = 5

rank(basis_size=basis_size,
    n_max = n_max,
    dim_max = dim_max)
