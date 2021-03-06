# ActSet
Active-Set solver for the non-negative LASSO problem.

## Non-negative LASSO

The non-negative LASSO is a famous convex problem finding applications in various fields. Starting from a given vector **y** and a given matrix **A**, it seeks for a sparse vector **x** such that **Ax** well approximates **y**. Many algorithms can tackle this problem very efficiently. Bests known are FISTA [1] and ADMM [2]. However, they face their limits in some cases. When the dimension of **y** is too large, iterations can be costly to handle. Furthermore, algorithms usually struggle to converge when a high accuracy is required. The Active-set method [3] seems to handle pretty well the problem in such cases.

[1] Beck, A., & Teboulle, M. (2009, April). A fast iterative shrinkage-thresholding algorithm with application to wavelet-based image deblurring. In 2009 IEEE International Conference on Acoustics, Speech and Signal Processing (pp. 693-696). IEEE.

[2] Wahlberg, B., Boyd, S., Annergren, M., & Wang, Y. (2012). An ADMM algorithm for a class of total variation regularized estimation problems. IFAC Proceedings Volumes, 45(16), 83-88.

[3] Nocedal, J., & Wright, S. (2006). Numerical optimization. Springer Science & Business Media.

## Usage

1. Import the solver and data-generation methods
```python
import numpy as np
from actset.dictionary import sample_data, sample_sparse_vector
from actset.actset import actset
```

2. Set problem parameters
```python
m = 100           # Number of lines of A
n = 200           # Number of columns of A
gen = 'uniform'   # Generation method for A
reg_ratio = 0.4   # Regularization ratio lambda / lambda_{max}
prec = 1e-10      # Target precision on the duality gap
maxit = int(1e4)  # Maximum number of iteration allowed
maxop = int(1e9)  # Maximum number of operation (multiplications) allowed
```

3. Generate data
```python
k = int(n/3)                                      # Sparsity parameter
sparse_vector = sample_sparse_vector(n, k)        # k-sparse vector used to generate A and y
A, y = sample_data(m, n, gen, sparse_vector)      # Data generation
reg = reg_ratio * np.linalg.norm(A.T @ y, np.inf) # Real regularization parameter
x0 = np.zeros(n)                                  # Initial point for the Active-set solver
```

4. Solve the problem and retrieve monitoring values
```python
x_actset, mnt_actset = actset(x0, A, y, reg, prec, maxit, maxop)
```
