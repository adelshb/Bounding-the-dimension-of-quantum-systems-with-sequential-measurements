import numpy as np
from basis_generator_nseq import generate_basis

import timeit

start = timeit.default_timer()


dim = 2
num_obs = 5
len_seq = 3
basis_size= 600

__, r = generate_basis(dim=dim,
                num_obs=num_obs,
                len_seq=len_seq,
                basis_size=basis_size)

stop = timeit.default_timer()
print('Time: ', stop - start)
print(r)

# dim = 2
# num_obs = 2
# len_seq = 3
# basis_size= 150
# Time:  1.559908428
# 7
#
# dim = 2
# num_obs = 3
# len_seq = 3
# basis_size= 150
# Time:  264.385502884
# 50

# dim = 2
# num_obs = 4
# len_seq = 3
# basis_size= 150
# Time:  2632.232994182
# 196
