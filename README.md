# Bounding the dimension of quantum systems via sequential projective measurements

 Bounding the dimension of quantum systems with sequential measurements. The optimizations are obtained with [MOSEK](https://www.mosek.com/) and can be changed whenever necessary

 ## Requirements
 * Python 3.6+
 * CVXPY 1.0
 * Seaborn

 ```shell
 pip install -r requirements.txt
 ```

 ### Example: Save the basis of moment matrices.

 ```shell
 python save_basis.py \
     --dim 2 \
     --num_obs 2 \
     --len_seq 2 \
     --num_out 2 \
     --stop 10000 \
     --save_metadata True \
     --save_data True
 ```

 ### Example: Compute the visibility of random moment matrices for a chosen dimension with given basis of moment matrices.

 ```shell
 python save_visibility.py \
     --num_obs 3 \
     --len_seq 2 \
     --num_out 2 \
     --dimX 3 \
     --data_samp 10 \
     --dim_base 2 \
     --level 1 \
     --basis_filename data_basis/2-dim-3-num_obs-2-len_seq-2-num_out.npy
 ```

 ### Example: Generate a random witness for between two specific dimensions in a chosen scenario

 ```shell
 python random_witness_generator.py \
     --num_obs 3 \
     --len_seq 2 \
     --num_out 2 \
     --dimX 3 \
     --num_samples 100 \
     --dim_base 2 \
     --remove_last_out True \
     --basis_filename data/data_basis/2-dim-3-num_obs-2-len_seq-2-num_out-1-level.npy
 ```
