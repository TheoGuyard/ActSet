# ActSet
Active-Set solver for the non-negative LASSO problem.

## Non-negative LASSO

The non-negative LASSO is a famous convex problem finding applications in various fields. It can be expressed as

![equation](http://www.sciweavers.org/upload/Tex2Img_1602350638/render.png)

for a given dictionary $\mathbf{A} \in \mathbb{R}^{m \times n}$, a given observation vector $\mathbf{y} \in \mathbb{R}^m$ and a given regularization parameter $\lambda \geq 0$. It seeks for a sparse vector $\mathbf{x}^{\star}$ approximating $\mathbf{y}$ through the coefficients of $\mathbf{A}$. The larger $\lambda$, the sparser $\mathbf{x}^{\star}$. Many algorithms can tackle this problem very efficiently. Bests known are FISTA [1] and ADMM [2]. However, they face their limits in some cases. When the dimension of $\mathbf{A}$ is too large, iterations can be costly to handle. Furthermore, algorithms usually struggle to converge when a high accuracy is required. Recently, screening tests [2] were developed to face the dimension issue by reducing iteratively the size of the problem during the resolution process. In this paper, we introduce new screening tests to tackle the non-negative LASSO when a high accuracy is required which allow to bypass the last iterations where methods are usually slow to converge.

## Usage

1. Import the solver and data-generation methods
```
import numpy as np
from actset.dictionary import sample_data, sample_sparse_vector
from actset.actset import actset
```

2. Set problem parameters
```
m = 100           # Number of lines of A
n = 200           # Number of columns of A
gen = 'uniform'   # Generation method for A
reg_ratio = 0.4   # Regularization ratio lambda / lambda_{max}
prec = 1e-10      # Target precision on the duality gap
maxit = int(1e4)  # Maximum number of iteration allowed
maxop = int(1e9)  # Maximum number of operation (multiplications) allowed
```

3. Generate data
```
k = int(n/3)
sparse_vector = sample_sparse_vector(n, k)        # k-sparse vector used to generate A and y
A, y = sample_data(m, n, gen, sparse_vector)      # Data generation
reg = reg_ratio * np.linalg.norm(A.T @ y, np.inf) # Real regularization parameter
x0 = np.zeros(n)                                  # Initial point for the Active-set solver
```

4. Solve the problem
```
x_actset, mnt_actset = actset(x0, A, y, reg, prec, maxit, maxop)
```
