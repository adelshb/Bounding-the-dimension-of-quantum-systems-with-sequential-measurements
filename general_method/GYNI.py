# Import packages.
import cvxpy as cp
import numpy as np
from basis_generator import generate_basis, sep_seq, all_seq

dim = 2
num_obs = 2
len_seq = 3
out_max = 1
basis_size= 50
#seq_method="sep_seq"
seq_method="all_sequences"
GYN_events = [((0, 0, 0), (0, 0, 0)),
            ((1, 1, 0), (0, 1, 1)),
            ((0, 1, 1), (1, 0, 1)),
            ((1, 0, 1), (1, 1, 0)),
                ]
# sequences = sep_seq(num_m=num_obs,
#             num_p=len_seq,
#             out_max=out_max)

sequences = all_seq(n=num_obs,
            r_max=len_seq,
            out_max=out_max)

#print(sequences)
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
