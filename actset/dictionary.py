import numpy as np
from scipy.linalg import toeplitz
from scipy.fftpack import dct

def sample_sparse_vector(n, k):
    x = np.zeros(n)
    nnidx = np.random.choice(range(n), k, replace=False)
    x[nnidx] = np.ones(k)
    return x

def _normalize_dict(A):
    for i in range(A.shape[1]):
        A[:, i] /= np.linalg.norm(A[:, i], 2)
    return A

def _normalize_obs(y):
    y /= np.linalg.norm(y, 2)
    return y

def _gaussian_dict(m, n):
    A = np.random.randn(n, m).T
    return A

def _gaussian_obs(m):
    y = np.random.randn(m)
    return y

def _uniform_dict(m, n):
    A = np.random.uniform(0, 1, (n, m)).T
    return A

def _uniform_obs(m):
    y = np.random.uniform(0, 1, m)
    return y

def _dct_dict(m, n):
    A = dct(np.eye(n))
    indices = np.random.permutation(n)
    indices = indices[:m]
    A = A[:, indices].T
    return A

def _dct_obs(m):
    y = np.cos([2 * i * np.pi / m for i in range(m)])
    return y

def _toeplitz_dict(m, n):

    """
    gauss = lambda t: np.exp(-.5 * (t**2))
    
    ranget = np.linspace(-10, 10, n)
    offset = 1.
    rangemu = np.linspace(np.min(ranget) + offset, np.max(ranget) - offset, m)

    A = np.zeros((n, m))

    for j in range(m):
        A[:, j] = gauss(ranget - rangemu[j])
    """

    A = toeplitz(np.random.rand(max(n, m)))
    A = A[:n, :m].T

    return A

def _constant_noise_dict(m, n):
    A = _gaussian_dict(n, m).T
    A += 5
    return A

def _constant_noise_obs(m):
    y = _gaussian_obs(m)
    y += 5
    return y

def sample_data(m, n, generation="gaussian", sparse_vector=None):
    """
    Generates a (m,n) matrix with l2-normalized columns using the generation 
    given in parameters. Matrix are first generated with size (n,m) and then 
    transposed in order to make column deletion faster. Observation vector y is
    either generated as Gaussian (if sparse_vector=None) or computed as 
    y = A @ sparse_vector and normalize.
    """
    if generation == "gaussian":
        A = _gaussian_dict(m, n)
    elif generation == "uniform":
        A = _uniform_dict(m, n)
    elif generation == "dct":
        A = _dct_dict(m, n)
    elif generation == "toeplitz":
        A = _toeplitz_dict(m, n)
    elif generation == "constant-noise":
        A = _constant_noise_dict(m, n)
    else:
        raise ValueError(f"{generation} generation type not understood")

    if sparse_vector is not None:
        y = A @ sparse_vector
    else:
        y = _gaussian_obs(m)
    
    A = _normalize_dict(A)
    y = _normalize_obs(y)

    return A, y
