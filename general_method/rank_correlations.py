from basis_generator import rand_moment
from numpy import linalg as LA
import numpy as np

dim=2
num_obs=3
len_seq=2
out_max=1
seq_method="all_sequences"

num = 20

Xs = [rand_moment(dim, num_obs, len_seq, out_max, seq_method=seq_method) for __ in range(num)]
Hs = [LA.eig(X)[1]@np.sqrt(np.diag(LA.eig(X)[0])) for X in Xs]

rankXs = [LA.matrix_rank(X) for X in Xs]
rankHs = [LA.matrix_rank(H) for H in Hs]

max_rXs = max(rankXs)
max_rHs = max(rankHs)
R = np.zeros((max_rXs,max_rHs))

ranks = list(zip(rankXs,rankHs))

for k in ranks:
    temp = {}
    temp[str(k)] = ranks.count(k)
    R.append()
