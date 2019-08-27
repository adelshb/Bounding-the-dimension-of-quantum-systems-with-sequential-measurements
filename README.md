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
python save_basis.py \
   --dim 2 \
   --num_obs 3 \
   --len_seq 2 \
   --basis_size 10 \
   --save True
 ```

 ### Example: Find the maximum violation of the Leggett-Garg inequality.

 ```shell
 python simplified_method/Leggett-Garg.py
 ```
