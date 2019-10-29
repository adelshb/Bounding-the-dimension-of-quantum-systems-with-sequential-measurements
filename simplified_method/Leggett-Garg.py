# Import packages.
import cvxpy as cp
import numpy as np

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
prob = cp.Problem(cp.Maximize(cp.trace(C@X)/2),
                  constraints)
prob.solve()

# Print result.
print("The optimal value is", prob.value)
#print("A solution X is")
#print(X.value)
