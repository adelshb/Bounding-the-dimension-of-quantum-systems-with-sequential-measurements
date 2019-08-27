import numpy as np

from argparse import ArgumentParser
import json
import timeit

from basis_generator_nseq import generate_basis, rank_basis

def main(args):

    dim = 2
    num_obs = 6
    len_seq = 2
    basis_size = 10
    meta_data = {}

    start = timeit.default_timer()

    X_basis = []
    Done = False
    while Done==False:
        X, __ = generate_basis(dim=dim,
                        num_obs=num_obs,
                        len_seq=len_seq,
                        basis_size=basis_size)
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

    meta_data["dimension"] = dim
    meta_data["maximum length of sequences"] = len_seq
    meta_data["num of observables"] = num_obs
    meta_data["time"] = stop - start
    meta_data["rank"] = int(rank_new)
    meta_data["number of elements"] = len(X_new_basis)

    dir_name = "data_basis/"
    NAME = '{}-dim-{}-num_obs-{}-len_seq'.format(dim, num_obs, len_seq)

    with open(dir_name + NAME + '-meta_data.json', 'w') as fp:
        json.dump(meta_data, fp)

    np.save(dir_name + NAME, X_new_basis)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Data parameters
    parser.add_argument("--dim", type=list, default=["W", "GHZ"])
    parser.add_argument("--data_size", type=int, default=100000)
    parser.add_argument("--num_qubits", type=int, default=3)
    parser.add_argument("--classification", type=str, default="SLOCC", choices=["LOCC", "SLOCC"])
    parser.add_argument("--sv_ratio_th", type=float, default=0.5)
    parser.add_argument("--method", type=str, default="random")

    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--shuffle", type=bool, default=True)

    args = parser.parse_args()
    main(args)
