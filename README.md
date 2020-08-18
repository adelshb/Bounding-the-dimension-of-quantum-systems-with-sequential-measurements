# Bounding the dimension of quantum systems via sequential measurements

 Bounding the dimension of quantum systems with sequential measurements.

 ## Requirements
 * Python 3.6+
 * CVXPY 1.0
 * Seaborn

 ```shell
 pip install -r requirements.txt
 ```
 ## Sets of moments matrices for different dimension in different scenarios:

 A scenario is defined by the number of measurements, the longest length of sequential measurements and the number of outcomes of the measurements.

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

 ### Example: Compute robustness of random moment matrices for chosen dimension with basis given a a specific level of the NPA hierarchy.

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

 ### Example: Find the maximum violation of the Leggett-Garg inequality.

 ```shell
 python public/Leggett-Garg.py
 ```
