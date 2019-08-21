import numpy as np
from basis_generator_nseq import generate_basis

dim = 2
num_obs = 5
len_seq = 2
basis_size=150

__, r = generate_basis(dim=dim,
                num_obs=num_obs,
                len_seq=len_seq,
                basis_size=basis_size)

print(r)

# dim = 3
# num_obs = 3
# len_seq = 2
# basis_size=200
#
# __, r = generate_basis(dim=dim,
#                 num_obs=num_obs,
#                 len_seq=len_seq,
#                 basis_size=basis_size)
#
# print(r)
