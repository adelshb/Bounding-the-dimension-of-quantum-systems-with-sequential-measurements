import numpy as np

import json

from basis_generator_nseq import generate_basis, rank_basis

dir_name = "data_basis/"

def load_basis(dim,
            num_obs,
            len_seq):

    dir_name = "data_basis/"
    NAME = '{}-dim-{}-num_obs-{}-len_seq'.format(dim, num_obs, len_seq)

    X_basis = np.load(dir_name + NAME + ".npy")

    return X_basis

dim = 2
num_obs = 6
len_seq = 2

X_basis = load_basis(dim,
                num_obs,
                len_seq)

print(len(X_basis))
