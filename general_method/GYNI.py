# Import packages.
import cvxpy as cp
import numpy as np
from basis_generator_nseq import generate_basis, sep_seq

dim = 2
num_obs = 2 # number of measurements per position!
len_seq = 3
out_max = 1
basis_size= 160
seq_method="sep_seq"
GYN_events = [((0, 0, 0), (0, 2, 4)),
            ((1, 1, 0), (0, 3, 5)),
            ((0, 1, 1), (1, 2, 5)),
            ((1, 0, 1), (1, 3, 4)),
                ]
sequences = sep_seq(num_m=num_obs,
            num_p=len_seq,
            out_max=out_max)

n = len(sequences)

X_basis, rank = generate_basis(dim, num_obs, len_seq, out_max, basis_size, seq_method=seq_method)
print(rank)

alpha = cp.Variable((len(X_basis), 1))

C = np.zeros((n+1,n+1))
for i, seq_row in enumerate(sequences):
    if seq_row in GYN_events:
        C[i+1,i+1] = 1

X = cp.Variable((n+1,n+1), symmetric=True)
constraints = [X >> 0]
constraints += [
    X[0,0] == 1
]
constraints += [
    X == sum([alpha[i]*X_basis[i] for i in range(len(X_basis))])
]
prob = cp.Problem(cp.Maximize(cp.trace(C@X)),
                  constraints)
prob.solve()

print(prob.status)
print("The optimal value is", prob.value)
