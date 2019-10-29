import numpy as np

from argparse import ArgumentParser
import multiprocessing

import json
import timeit

from basis_generator import generate_basis, rank_basis

# Example:
# python save_basis_multi-cores.py \
#     --dim 2 \
#     --num_obs 3 \
#     --len_seq 2 \
#     --out_max 1 \
#     --batch_init 100 \
#     --batch_size 20 \
#     --dtype float16
#     --save False

def main(args):

    CPUs = multiprocessing.cpu_count()
    input = [(args.dim, args.num_obs, args.len_seq, args.out_max-1, 1, "all_sequences", False)]
    pool = multiprocessing.Pool(processes = CPUs)

    start = timeit.default_timer()
    X_basis = []
    Done = False
    init = True
    while Done==False:

        if init == True:
            X = pool.starmap(generate_basis, input*args.batch_init)
        else:
            X = pool.starmap(generate_basis, input*args.batch_size)
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

    stop = timeit.default_timer()

    if args.save == True:
        meta_data = {}
        meta_data["dimension"] = args.dim
        meta_data["maximum length of sequences"] = args.len_seq
        meta_data["num of observables"] = args.num_obs
        meta_data["num of outcomes"] = args.out_max
        meta_data["time"] = stop - start
        meta_data["rank"] = int(rank_new)
        meta_data["number of elements"] = len(X_new_basis)
        meta_data["moment size"] = X_new_basis[0].shape
        meta_data["dtype"] = args.dtype

        dir_name = "data_basis/"
        NAME = '{}-dim-{}-num_obs-{}-len_seq-{}-out_max'.format(args.dim, args.num_obs, args.len_seq, args.out_max)

        with open(dir_name + NAME + '-meta_data.json', 'w') as fp:
            json.dump(meta_data, fp)

        np.save(dir_name + NAME, [X.astype(np.dtype(args.dtype), copy=False) for X in X_new_basis])

    print("The running time is {}".format(stop - start))
    print("The rank is {}".format(rank_new))
    print("Done!")

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

    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--num_obs", type=int, default=3)
    parser.add_argument("--len_seq", type=int, default=2)
    parser.add_argument("--out_max", type=int, default=2)
    parser.add_argument("--batch_init", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--save", type=str2bool, nargs='?',
                        const=True, default=False)

    args = parser.parse_args()
    main(args)
