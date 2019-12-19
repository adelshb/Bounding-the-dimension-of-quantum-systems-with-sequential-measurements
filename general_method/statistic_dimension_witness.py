import numpy as np
import cvxpy as cp
from basis_generator import rand_moment, generate_basis
from argparse import ArgumentParser

_available_sequences_methods = [
    "all_sequences",
    "sep_seq",
    "sel_sequences",
    ]

def main(args):

    #Generate random moment matrices
    X = rand_moment(dim_behavior, num_obs, len_seq, out_max, seq_method, remove_last_out=remove_last_out)

    #Generate set of moment matrix at specific level of the hierarchy
    X_basis, __ = generate_basis(dim = dim_basis,
                                  num_obs = num_obs,
                                  len_seq= len_seq + level - 1,
                                  out_max = out_max,
                                  batch_size = batch_size,
                                  seq_method=seq_method,
                                  sel_sequences = [len_seq, len_seq + level - 1],
                                  remove_last_out=remove_last_out)

    #Compute the visibility
    eta = cp.Variable((1, 1))
    alpha = cp.Variable((len(X_basis), 1), complex=True)
    Gamma = cp.Variable(X_basis[0].shape)

    constraints = [Gamma >> 0]
    constraints += [
        Gamma == sum([alpha[i]*X_basis[i] for i in range(len(X_basis))])
    ]
    for i in range(1,len(X)):
        constraints += [
            eta*X[i,i] + (1-eta)/2 == Gamma[i,i]
        ]

    prob = cp.Problem(cp.Maximize(eta),
                      constraints)

    prob.solve(verbose=True)

    return prob.value

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--dim_behavior", type=int, default=3)
    parser.add_argument("--dim_basis", type=int, default=2)
    parser.add_argument("--num_obs", type=int, default=3)
    parser.add_argument("--len_seq", type=int, default=2)
    parser.add_argument("--num_out", type=int, default=2)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--sequence_method", type=str, default="sel_sequences", choices=_available_sequences_methods)

    args = parser.parse_args()
    main(args)
