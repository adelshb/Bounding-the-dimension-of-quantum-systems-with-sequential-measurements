# Import packages.
import cvxpy as cp
import numpy as np
from basis_generator import generate_basis

dim = 10
sequences = [[0,1],[0,2],[1,2]]
X_basis = generate_basis(dim, sequences)

alpha = cp.Variable((len(X_basis),1))

n = 3
# Generate a random SDP.
C = np.array([
        [0, 1, -1],
        [1, 0, 1],
        [-1, 1, 0]
])

# Define and solve the CVXPY problem.
# Create a symmetric matrix variable.
X = cp.Variable((n,n), symmetric=True)
# The operator >> denotes matrix inequality.
constraints = [X >> 0]
constraints += [
    X[i,i] == 1 for i in range(n)
]
constraints += [
    X == sum([alpha[i]*X_basis[i] for i in range(len(X_basis))])
]
prob = cp.Problem(cp.Maximize(cp.trace(C@X)/2),
                  constraints)
prob.solve()

# Print result.
print("The optimal value is", prob.value)
print("A solution X is")
print(X.value)
