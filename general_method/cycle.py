# Import packages.
import cvxpy as cp
import numpy as np
from basis_generator_nseq import generate_basis, all_seq

dim = 2
num_obs = 7
len_seq = 2
out_max = 1
basis_size= 300 #
C_events = []

for i in range(num_obs):
    C_events.append(((0, 1), (i, i+1)))
    C_events.append(((1, 0), (i, i+1)))

C_events.append(((0, 1), (num_obs-1, 0)))
C_events.append(((1, 0), (num_obs-1, 0)))

sequences = all_seq(num_obs, r_max=len_seq, out_max=out_max)
n = len(sequences)

X_basis, rank = generate_basis(dim, num_obs, len_seq, out_max, basis_size)
print(rank)

alpha = cp.Variable((len(X_basis), 1))

C = np.zeros((n+1,n+1))
for i, seq_row in enumerate(sequences):
    if seq_row in C_events:
        C[i+1,i+1] = 1

X = cp.Variable((n+1,n+1), symmetric=True)
constraints = [X >> 0]
constraints += [
    X[0,0] == 1
]
constraints += [
    X == sum([alpha[i]*X_basis[i] for i in range(len(X_basis))])
]
prob = cp.Problem(cp.Minimize(num_obs-2*cp.trace(C@X)),
                  constraints)
prob.solve()

# Print result.
print(prob.status)
print("The optimal value is", prob.value)
#print("A solution X is")
#print(X.value)
#print(alpha.value)
