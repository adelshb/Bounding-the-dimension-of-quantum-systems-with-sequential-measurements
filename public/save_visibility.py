import numpy as np
from numpy import linalg as LA
import cvxpy as cp
import scipy

from argparse import ArgumentParser
import multiprocessing
import json
import timeit

from basis_generator import rand_moment

# Example:
# python save_visibility.py \
#     --num_obs 3 \
#     --len_seq 2 \
#     --num_out 2 \
#     --dimX 3 \
#     --data_samp 10 \
#     --dim_base 2 \
#     --level 1 \
#     --basis_filename data_basis/2-dim-3-num_obs-2-len_seq-2-num_out.npy

def main(args):

    start = timeit.default_timer()

    print("Parameters: {}.".format(args))

    X_basis = np.load(args.basis_filename)
    # Etas = []
    # for __ in range(args.data_samp):
    #     X = rand_moment(args.dimX,
    #                  args.num_obs,
    #                  args.len_seq,
    #                  args.num_out,
    #                  sel_sequences = [args.len_seq, args.len_seq+args.level-1],
    #                  remove_last_out = True)
    #
    #     Etas.append(single_behavior_visibility(X, X_basis))

    CPUs = multiprocessing.cpu_count()
    print("Number of CPUs: {}.".format(CPUs))
    input = [(args.dimX, args.num_obs, args.len_seq, args.num_out, args.level, X_basis)]
    pool = multiprocessing.Pool(processes = CPUs)
    Etas = pool.starmap(rand_moment2rand_vis, input*args.data_samp)

    data = {}
    data["num of observables"] = args.num_obs
    data["maximum length of sequences"] = args.len_seq
    data["num of outcomes"] = args.num_out
    data["dimension behaviors"] = args.dimX
    data["dimension base"] = args.dim_base
    data["level"] = args.level
    data["visibilities"] = Etas

    NAME = '{}-num_obs-{}-len_seq-{}-out_max-{}-dim_behavior-{}-dim_base-{}-level'.format(args.num_obs, args.len_seq, args.num_out, args.dimX, args.dim_base, args.level)

    with open("data_robustness/" + NAME + '.json', 'w') as fp:
        json.dump(data, fp, indent=2)

    print("Number of moment matrices generated and tested: {}.".format(len(Etas)))
    stop = timeit.default_timer()
    print("Running time: {}.".format(stop - start))
    print("Done!")
    return

def rand_moment2rand_vis(dimX, num_obs, len_seq, num_out, level, X_basis):

    scipy.random.seed()
    X = rand_moment(dimX,
                 num_obs,
                 len_seq,
                 num_out,
                 sel_sequences = [len_seq, len_seq+level-1],
                 remove_last_out = True)

    return single_behavior_visibility(X, X_basis)

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
    return eta.value[0][0]

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--num_obs", type=int, default=3)
    parser.add_argument("--len_seq", type=int, default=2)
    parser.add_argument("--num_out", type=int, default=2)
    parser.add_argument("--dimX", type=int, default=3)
    parser.add_argument("--data_samp", type=int, default=100)
    parser.add_argument("--dim_base", type=int, default=2)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--basis_filename", type=str, default="data_basis/2-dim-3-num_obs-2-len_seq-2-num_out.npy")

    args = parser.parse_args()
    main(args)
