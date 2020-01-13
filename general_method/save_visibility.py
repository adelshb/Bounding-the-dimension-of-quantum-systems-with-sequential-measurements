import numpy as np
import cvxpy as cp

from argparse import ArgumentParser
import multiprocessing
import json

from basis_generator import rand_moment, generate_basis, sel_seq, rank_basis

def main(args):

    print("Parameters: ", args)
    dir_name = "data/"

    for k in range(1,args.level_max+1):

        print("Level: ", k)
        etak, rankk = visibility(num_obs=args.num_obs,
                            len_seq=args.len_seq,
                            out_max=args.out_max,
                            seq_method= args.seq_method,
                            dimX= args.dimX,
                            data_samp = args.data_samp,
                            dim_base= args.dim_base,
                            level=k,
                            batch_size= args.batch_size*k**2)

        print(etak)
        print(rankk)

        data = {}
        data["num of observables"] = args.num_obs
        data["maximum length of sequences"] = args.len_seq
        data["num of outcomes"] = args.out_max +1
        data["sequences method"] = args.seq_method
        data["dimension behaviors"] = args.dimX
        data["dimension base"] = args.dim_base
        data["level"] = k
        data["visibilities"] = etak
        data["base ranks"] = int(rankk)

        print(data)

        NAME = '{}-num_obs-{}-len_seq-{}-out_max-{}-dim_behavior-{}-dim_base-{}-level'.format(args.num_obs, args.len_seq, args.out_max, args.dimX, args.dim_base, k)

        with open(dir_name + NAME + '.json', 'w') as fp:
            json.dump(data, fp, indent=2)

    print("Done.")
    return

def visibility(num_obs=3,
            len_seq=2,
            out_max=1,
            seq_method= "sel_sequences",
            dimX=3,
            data_samp = 10,
            dim_base=2,
            level=1,
            batch_size=50):

    sequences = sel_seq(n = num_obs,
                    r = len_seq,
                    out_max = out_max,
                    remove_last_out = True)

    if level > 1:
        hierarchy_seq = sequences + sel_seq(n = num_obs,
                                        r = len_seq + level -1,
                                        out_max = out_max,
                                        remove_last_out = True)
    else:
        hierarchy_seq = sequences

    CPUs = multiprocessing.cpu_count()
    input = [(dim_base, num_obs, len_seq, out_max, 1, 'sel_sequences', [len_seq, len_seq + level - 1], True, False)]
    pool = multiprocessing.Pool(processes = CPUs)

    X_basis = []
    Done = False
    while Done==False:

        X = pool.starmap(generate_basis, input*batch_size)
        X = sum(X,[])

        X_basis = X_basis + X
        rank = rank_basis(X_basis)
        if rank < len(X_basis):
            Done = True

    Done = False
    extra = 0
    while Done==False:
        X_new_basis = X_basis[:rank+extra]
        rank_new = rank_basis(X_new_basis)
        if rank_new == rank:
            Done = True
        extra += 1

    input = [(dimX, num_obs, len_seq, out_max, 'sel_sequences', X_basis)]

    etas = pool.starmap(single_behavior_visibility, input*data_samp)
    etas = list(filter(None, etas))

    return etas, rank

def single_behavior_visibility(dimX, num_obs, len_seq, out_max, seq_method, X_basis):

    X = rand_moment(dimX, num_obs, len_seq, out_max, seq_method, [len_seq] ,remove_last_out=True)

    eta = cp.Variable((1, 1))
    alpha = cp.Variable((len(X_basis), 1))

    constraints = [sum([alpha[j]*X_basis[j] for j in range(len(X_basis))]) >> 0]
    constraints += [sum([alpha[j]*X_basis[j] for j in range(len(X_basis))])[0,0] == 1]
    for i in range(1,len(X)):
        constraints += [
            #eta*X[i,i] + (1-eta)*X_basis_2d[0][i,i] == sum([alpha[j]*X_basis_2d[j][i,i] for j in range(len(X_basis_2d))])
            eta*X[i,i] + (1-eta)/2 == sum([alpha[j]*X_basis[j][i,i] for j in range(len(X_basis))])
        ]

    prob = cp.Problem(cp.Maximize(eta),
                      constraints)
    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
        return eta.value[0][0]
    except:
        return None


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--num_obs", type=int, default=3)
    parser.add_argument("--len_seq", type=int, default=2)
    parser.add_argument("--out_max", type=int, default=1)
    parser.add_argument("--seq_method", type=str, default="sel_sequences")
    parser.add_argument("--dimX", type=int, default=3)
    parser.add_argument("--data_samp", type=int, default=50)
    parser.add_argument("--dim_base", type=int, default=2)
    parser.add_argument("--level_max", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=50)

    args = parser.parse_args()
    main(args)
