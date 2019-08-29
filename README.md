# Bounding the dimension of quantum systems via sequential measurements

 Bounding the dimension of quantum systems with sequential measurements.

 ## Requirements
 * Python 3.6+
 * CVXPY 1.0

 ```shell
 pip install -r requirements.txt
 ```

 ### Example: Save the basis of moment matrices.

 ```shell
python save_basis_multi-cores.py \
   --dim 2 \
   --num_obs 3 \
   --len_seq 2 \
   --out_max 1 \
   --batch_init 100 \
   --basis_size 20 \
   --save True
 ```

 ### Example: Find the maximum violation of the Leggett-Garg inequality.

 ```shell
 python simplified_method/Leggett-Garg.py
 ```
