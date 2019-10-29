# Bounding the dimension of quantum systems via sequential measurements

 Bounding the dimension of quantum systems with sequential measurements.

 ## Requirements
 * Python 3.6+
 * CVXPY 1.0

 ```shell
 pip install -r requirements.txt
 ```
 ## Sets of moments matrices for different dimension in different scenarios:

 A scenario is defined by the number of measurements, the longest length of sequential measurements and the number of outcomes of the measurements.

 | Dimension | Number of measurements | Maximum length of sequences | Number of outcomes | Rank |
 | --------- | ---------------------- | --------------------------- | ------------------ | ---- |
 | 2  | 2 | 3 | 2 | 10 |
 | 2  | 3 | 2 | 2 | 22 |
 | 2  | 3 | 3 | 2 | 50 |
 | 2  | 3 | 4 | 2 | 95 |
 | 2  | 4 | 2 | 2 | 56 |
 | 2  | 5 | 2 | 2 | 121 |
 | 2  | 6 | 2 | 2 | 232 |
 | 3  | 3 | 2 | 2 | 28 |
 | 3  | 3 | 2 | 3 | 271 |
 | 3  | 3 | 3 | 2 | 106 |
 | 3  | 3 | 3 | 3 | 1435 |
 | 3  | 4 | 2 | 2 | 89 |
 | 3  | 4 | 2 | 3 | 1065 |
 | 3  | 5 | 2 | 2 | 226 |
 | 3  | 6 | 2 | 2 | 487 |
 | 4  | 3 | 2 | 2 | 28 |
 | 4  | 3 | 2 | 3 | 271 |
 | 4  | 3 | 3 | 2 | 106 |
 | 4  | 3 | 3 | 3 | 1878 |
 | 4  | 4 | 2 | 2 | 89 |
 | 4  | 4 | 2 | 3 | 1065 |
 | 4  | 5 | 2 | 2 | 226 |
 | 4  | 6 | 2 | 2 | 487 |
 | 5  | 3 | 2 | 3 | 271 |
 | 5  | 4 | 2 | 3 | 1065 |

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
