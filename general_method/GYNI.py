# Import packages.
import cvxpy as cp
import numpy as np
from basis_generator import generate_basis, sep_seq, all_seq, sel_seq

dim = 2
num_obs = 2
len_seq = 3
out_max = 1
basis_size= 20
seq_method="sel_sequences"

GYN_events = [((0, 0, 0), (0, 0, 0)),
            ((1, 1, 0), (0, 1, 1)),
            ((0, 1, 1), (1, 0, 1)),
            ((1, 0, 1), (1, 1, 0)),
                ]

sequences = sel_seq(n = num_obs,
                r = len_seq,
                out_max = out_max,
                remove_last_out = False)

n = len(sequences)

X_basis, rank = generate_basis(dim, num_obs, len_seq, out_max, basis_size, seq_method=seq_method, sel_sequences = [len_seq], remove_last_out = False)
print("Basis rank: ", rank)

alpha = cp.Variable((len(X_basis), 1))

C = np.zeros((n,n))
for i, seq_row in enumerate(sequences):
    if seq_row in GYN_events:
        C[i,i] = 1

X = cp.Variable((n,n), symmetric=True)
#constraints = [cp.sum([alpha[j]*X_basis[j] for j in range(len(X_basis))]) >> 0]
constraints = [X >> 0]
# constraints += [
#     X[0,0] == 1
# ]
constraints += [
    X == sum([alpha[i]*X_basis[i] for i in range(len(X_basis))])
]
prob = cp.Problem(cp.Maximize(cp.trace(C@X)),
                  constraints)
prob.solve()

print(prob.status)
print("The optimal value is", prob.value)
