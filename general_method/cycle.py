# Import packages.
import cvxpy as cp
import numpy as np
from basis_generator import generate_basis, all_seq

def cycle_dim_SDP(dim,
                num_obs,
                basis_size,
                len_seq=2,
                out_max=1):

    C_events = []

    for i in range(num_obs):
        C_events.append(((0, 1), (i, i+1)))
        C_events.append(((1, 0), (i, i+1)))

    if num_obs % 2 == 0:
        C_events.append(((0, 0), (num_obs-1, 0)))
        C_events.append(((1, 1), (num_obs-1, 0)))
    else:
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
            # if num_obs % 2 == 0 and seq_row == C_events[-1]:
            #     C[i+1,i+1] = -1
            # elif num_obs % 2 == 0 and seq_row == C_events[-2]:
            #     C[i+1,i+1] = -1

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

    return prob.value, prob.status

def cycle_simp_SDP(num_obs,
            len_seq=2,
            out_max=1):

    C = np.zeros((num_obs,num_obs))
    for i in range(num_obs-1):
        C[i,i+1] = 1 # <A_i A_i+1> for i < N

    if num_obs % 2 == 0:
        C[0,num_obs-1] = -1 #<A_1 A_N>
    else:
        C[0,num_obs-1] = 1 #<A_1 A_N>

    X = cp.Variable((num_obs,num_obs), symmetric=True)
    constraints = [X >> 0]
    constraints += [
        X[i,i] == 1 for i in range(num_obs)
    ]
    prob = cp.Problem(cp.Minimize(cp.trace(C@X)),
                      constraints)
    prob.solve()

    return prob.value, prob.status

dim = 2
num_obs = 5
basis_size= 150# empirical

val, status = cycle_dim_SDP(dim,
            num_obs,
            basis_size,
            len_seq=2,
            out_max=1)

print(val)
print(status)


val, status = cycle_simp_SDP(num_obs,
            len_seq=2,
            out_max=1)

print(val)
print(status)
