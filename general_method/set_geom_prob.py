import numpy as np
import cvxpy as cp
from general_method.basis_generator import rand_moment, generate_basis, sel_seq


def visibility(num_obs=3,
            len_seq=2,
            out_max=1,
            seq_method= "sel_sequences",
            remove_last_out= True,
            dimX=3,
            dim_base=2,
            level=1,
            batch_size=50,
            verbose = False):

    X = rand_moment(dimX, num_obs, len_seq, out_max, seq_method, [len_seq] ,remove_last_out=remove_last_out)

    sequences = sel_seq(n = num_obs,
                    r = len_seq,
                    out_max = out_max,
                    remove_last_out = remove_last_out)

    if level > 1:
        hierarchy_seq = sequences + sel_seq(n = num_obs,
                                        r = len_seq + level -1,
                                        out_max = out_max,
                                        remove_last_out = remove_last_out)
    else:
        hierarchy_seq = sequences


    X_basis_2d, rank = generate_basis(dim = dim_base,
                                      num_obs = num_obs,
                                      len_seq= len_seq + level - 1,
                                      out_max = out_max,
                                      batch_size = batch_size,
                                      seq_method=seq_method,
                                      sel_sequences = [len_seq, len_seq + level - 1],
                                      remove_last_out=remove_last_out)


    eta = cp.Variable((1, 1))
    alpha = cp.Variable((len(X_basis_2d), 1))

    constraints = [sum([alpha[j]*X_basis_2d[j] for j in range(len(X_basis_2d))]) >> 0]
    constraints += [sum([alpha[j]*X_basis_2d[j] for j in range(len(X_basis_2d))])[0,0] == 1]
    for i in range(1,len(X)):
        constraints += [
            #eta*X[i,i] + (1-eta)*X_basis_2d[0][i,i] == sum([alpha[j]*X_basis_2d[j][i,i] for j in range(len(X_basis_2d))])
            eta*X[i,i] + (1-eta)/2 == sum([alpha[j]*X_basis_2d[j][i,i] for j in range(len(X_basis_2d))])

        ]

    prob = cp.Problem(cp.Maximize(eta),
                      constraints)

    prob.solve(solver=cp.MOSEK, verbose=verbose)

    return X, eta.value


def visibility_stability(num_obs=3,
                        len_seq=2,
                        out_max=1,
                        seq_method= "sel_sequences",
                        remove_last_out= True,
                        dimX=3,
                        dim_base=2,
                        level=1,
                        batch_size=50,
                        verbose = False,
                        eta_size = 10):

    X = rand_moment(dimX, num_obs, len_seq, out_max, seq_method, [len_seq] ,remove_last_out=remove_last_out)

    sequences = sel_seq(n = num_obs,
                    r = len_seq,
                    out_max = out_max,
                    remove_last_out = remove_last_out)

    if level > 1:
        hierarchy_seq = sequences + sel_seq(n = num_obs,
                                        r = len_seq + level -1,
                                        out_max = out_max,
                                        remove_last_out = remove_last_out)
    else:
        hierarchy_seq = sequences

    data_eta = []
    for __ in range(eta_size):
        X_basis_2d, _ = generate_basis(dim = dim_base,
                                          num_obs = num_obs,
                                          len_seq= len_seq + level - 1,
                                          out_max = out_max,
                                          batch_size = batch_size,
                                          seq_method=seq_method,
                                          sel_sequences = [len_seq, len_seq + level - 1],
                                          remove_last_out=remove_last_out)


        eta = cp.Variable((1, 1))
        alpha = cp.Variable((len(X_basis_2d), 1))

        constraints = [sum([alpha[j]*X_basis_2d[j] for j in range(len(X_basis_2d))]) >> 0]
        constraints += [sum([alpha[j]*X_basis_2d[j] for j in range(len(X_basis_2d))])[0,0] == 1]
        for i in range(1,len(X)):
            constraints += [
                #eta*X[i,i] + (1-eta)*X_basis_2d[0][i,i] == sum([alpha[j]*X_basis_2d[j][i,i] for j in range(len(X_basis_2d))])
                eta*X[i,i] + (1-eta)/2 == sum([alpha[j]*X_basis_2d[j][i,i] for j in range(len(X_basis_2d))])

            ]

        prob = cp.Problem(cp.Maximize(eta),
                          constraints)

        prob.solve(solver=cp.MOSEK, verbose=verbose)

        data_eta.append(eta.value[0][0])

    return X, data_eta
