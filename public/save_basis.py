import numpy as np

from argparse import ArgumentParser
import json
import timeit

from basis_generator import basis_gs

# Example:
# python save_basis.py \
#     --dim 2 \
#     --num_obs 2 \
#     --len_seq 2 \
#     --num_out 2 \
#     --norm_prec 0.0000001 \
#     --stop 10000 \
#     --save_metadata True \
#     --save_data True

def main(args):

    start = timeit.default_timer()

    X_basis = basis_gs(args.dim,
                        args.num_obs,
                        args.len_seq,
                        args.num_out,
                        args.remove_last_out,
                        #args.prec,
                        args.stop)

    stop = timeit.default_timer()

    if args.save_metadata == True:
        meta_data = {}
        meta_data["dimension"] = args.dim
        meta_data["maximum length of sequences"] = args.len_seq
        meta_data["num of observables"] = args.num_obs
        meta_data["num of outcomes"] = args.num_out
        meta_data["time"] = stop - start
        meta_data["number of LI moment matrices"] = len(X_basis)
        meta_data["moment matrix size"] = X_basis[0].shape
        meta_data["level"] = args.level
        meta_data["norm precision"] = args.norm_prec

        dir_name = "data_basis/"
        NAME = '{}-dim-{}-num_obs-{}-len_seq-{}-num_out'.format(args.dim, args.num_obs, args.len_seq, args.num_out)

        with open(dir_name + NAME + '-meta_data.json', 'w') as fp:
            json.dump(meta_data, fp)

    if args.save_data == True:
        np.save(dir_name + NAME, [X for X in X_basis])

    print("The running time is {}".format(stop - start))
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
    parser.add_argument("--num_out", type=int, default=2)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--remove_last_out", type=str2bool, nargs='?',
                        const=True, default=True)
    #parser.add_argument("--norm_prec", type=float, default=0.0000001)
    parser.add_argument("--stop", type=int, default=10000)

    parser.add_argument("--save_metadata", type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument("--save_data", type=str2bool, nargs='?',
                        const=True, default=True)

    args = parser.parse_args()
    main(args)
