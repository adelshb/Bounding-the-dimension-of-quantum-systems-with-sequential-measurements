import numpy as np
import cvxpy as cp
from basis_generator import rand_moment
from argparse import ArgumentParser
import json
import timeit

# Example:
# python random_witness_generator.py \
#     --num_obs 3 \
#     --len_seq 2 \
#     --num_out 2 \
#     --dimX 3 \
#     --num_samples 100 \
#     --dim_base 2 \
#     --remove_last_out True \
#     --basis_filename data/data_basis/2-dim-3-num_obs-2-len_seq-2-num_out-1-level.npy

def main(args):

    start = timeit.default_timer()

    print("Parameters: {}.".format(args))

    X_basis = np.load(args.basis_filename)

    X, rho, P = rand_moment(args.dimX, args.num_obs, args.len_seq, args.num_out, [args.len_seq], args.remove_last_out)
    eta, coef = single_behavior_visibility(X, X_basis)

    for __ in range(args.num_samples):
        X_t, rho_t, P_t = rand_moment(args.dimX, args.num_obs, args.len_seq, args.num_out, [args.len_seq], args.remove_last_out)
        eta_t, coef_t = single_behavior_visibility(X_t, X_basis)
        if eta_t < eta:
            eta = eta_t
            coef = coef_t
            X = X_t
            rho = rho_t
            P = P_t

    Q = np.real(np.diag(X)[1:] @ coef)
    C = NPA_bound(coef,X_basis)

    print("visibility: ", eta)
    print("Value of the witness for the generated behavior: ", Q)
    print("Maximum value of the witness for characterized set: ", C)

    data = {}
    data["Inequality"] = coef
    data["Classical Bound"] = C
    data["Quantum Bound"] = Q

    data["num of observables"] = args.num_obs
    data["maximum length of sequences"] = args.len_seq
    data["num of outcomes"] = args.num_out
    data["dimension behavior"] = args.dimX
    data["dimension base"] = args.dim_base

    NAME = '{}-num_obs-{}-len_seq-{}-out_max-{}-dim_behavior-{}-dim_base'.format(args.num_obs, args.len_seq, args.num_out, args.dimX, args.dim_base)

    with open("data/dimension_witness/" + NAME + '.json', 'w') as fp:
        json.dump(data, fp, indent=2)

    np.save("data/dimension_witness/" + NAME + "-moment_matrix", X)
    np.save("data/dimension_witness/" + NAME + "-state", rho)
    np.save("data/dimension_witness/" + NAME + "-measurements", P)


    stop = timeit.default_timer()
    print("Running time: {}.".format(stop - start))
    print("Done!")
    return

def single_behavior_visibility(X, X_basis):

    eta = cp.Variable((1, 1))
    alpha = cp.Variable((len(X_basis), 1))
    beta = cp.Variable((len(X_basis), 1))
    M = cp.Variable(X_basis[0].shape)
    N = cp.Variable(X_basis[0].shape)

    constraints = [N >> 0]
    constraints += [N == sum([beta[j]*X_basis[j] for j in range(len(X_basis))])]
    constraints += [N[0,0] == 1 - eta]

    for i in range(1,len(X)):
        constraints += [
            eta*X[i,i] + N[i,i] == M[i,i]
        ]

    constraints += [M >> 0]
    constraints += [M == sum([alpha[j]*X_basis[j] for j in range(len(X_basis))])]
    constraints += [M[0,0] == 1]

    prob = cp.Problem(cp.Maximize(eta),
                      constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)

    coef = []
    for __ in range(3,len(constraints)-3):
        coef.append(np.real(constraints[__].dual_value[0][0]))

    return eta.value[0][0], coef

def NPA_bound(coef,X_basis):
    beta = cp.Variable((len(X_basis), 1))
    N = cp.Variable(X_basis[0].shape)

    constraints = [N >> 0]
    constraints += [N == sum([beta[j]*X_basis[j] for j in range(len(X_basis))])]
    constraints += [N[0,0] == 1]

    prob = cp.Problem(cp.Minimize(
            sum([N[j+1,j+1]*coef[j] for j in range(len(coef))])
            ),constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)
    return prob.value

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--num_obs", type=int, default=3)
    parser.add_argument("--len_seq", type=int, default=2)
    parser.add_argument("--num_out", type=int, default=2)
    parser.add_argument("--dimX", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--dim_base", type=int, default=2)
    parser.add_argument("--remove_last_out", type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument("--basis_filename", type=str, default="data/data_basis/2-dim-3-num_obs-2-len_seq-2-num_out.npy")

    args = parser.parse_args()
    main(args)
