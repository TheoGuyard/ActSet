import numpy as np
import matplotlib.pyplot as plt
from actset.dictionary import sample_data, sample_sparse_vector
from actset.actset import actset

m, n, gen = 100, 200, 'uniform'
reg_ratio = 0.4
sparse_vector = sample_sparse_vector(n, int(n/3))
A, y = sample_data(m, n, gen, sparse_vector)
reg = reg_ratio * np.linalg.norm(A.T @ y, np.inf)
x0 = np.zeros(n)
prec, maxit, maxop = 1e-10, int(1e4), int(1e9)

x_actset, mnt_actset = actset(x0, A, y, reg, prec, maxit, maxop)
print("Convergence :", mnt_actset['conv'])

fig, ax = plt.subplots(2, 1)

ax[0].plot(np.cumsum(mnt_actset['mult']), mnt_actset['gap'], label='Active-set', marker='.')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel('Operations')
ax[0].set_ylabel('Dual gap')

ax[1].plot(np.cumsum(mnt_actset['mult']), n - mnt_actset['w_size'], label='Active-set', marker='.')
ax[1].set_xscale('log')
ax[1].set_xlabel('Operations')
ax[1].set_ylabel('Coefficients not in the working set')

plt.tight_layout()
plt.show()