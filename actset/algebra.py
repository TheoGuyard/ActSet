import numpy as np
from scipy.linalg import cholesky


def cholesky_inverse(X, tri=False):
    """
    Computes (X)^{-1} where X is a positive definite matrix using Cholesky 
    factorization. If tri=True, only Linv is returned
    """

    mult = 0
    n = X.shape[0]
    L = cholesky(X, lower=True)
    Linv = np.linalg.inv(L)

    # Operations :
    #   - L         : 1/3 * n * (n**2 - 1) multiplications
    #   - Linv      : n * (n-1) / 2 multiplications
    mult += n * (n ** 2 - 1) / 3 + n * (n - 1) / 2

    if tri:
        return Linv, mult
    else:
        Xinv = Linv.T @ Linv

        # Operations :
        #   - Xinv      : n**3 multiplications
        mult += n ** 3

        return Xinv, mult


def gram_cholesky_inverse(X):
    """
    Computes (XtX)^{-1} from a (m,n) matrix (with n<=m) using Cholesky 
    factorization.
    """

    m, n = X.shape
    L = cholesky(X.T @ X, lower=True)
    Linv = np.linalg.inv(L)
    Graminv = Linv.T @ Linv
    # Operations :
    #   - X.T @ X   : n * m multiplications
    #   - L         : 1/3 * n * (n**2 - 1) multiplications
    #   - Linv      : n * (n - 1) / 2 multiplications
    #   - Graminv    : n**3 multiplications
    mult = n * m + n * (n ** 2 - 1) / 3 + n * (n - 1) + n ** 3

    return Graminv, mult


def downdate_inverse(Graminv, idx):
    """
    Given a matrix Graminv = (XtX)^{-1}, computes the downdated Graminv when 
    the columns at indexes idx are removes from X.
    """

    mult = 0
    idx = np.sort(idx)

    for i in idx:
        n = Graminv.shape[0]
        q = Graminv[i, i]
        c = np.delete(np.arange(Graminv.shape[0]), i)
        u = Graminv[c, i]
        Graminv = np.delete(Graminv, i, axis=0)
        Graminv = np.delete(Graminv, i, axis=1)
        Graminv -= np.outer(u, u / q)
        idx -= 1

        # Operations :
        #   - u/q           : n multiplications
        #   - outer(u, u/q) : n**2 multiplications
        mult += n ** 2 + n

    return Graminv, mult


def update_inverse(Graminv, X, idx, c):
    """
    Given a matrix Graminv = (XtX)^{-1}, computes the updated Graminv when the 
    column c is added to X at index idx.
    """

    m, n = X.shape
    u1 = X.T @ c
    u2 = Graminv @ u1
    d = 1 / (c @ c - u1 @ u2)
    u3 = d * u2
    F11m1 = Graminv + np.outer(u3, u2)

    # Operations :
    #   - u1    : n * m  multiplications
    #   - u2    : n**2 multiplications
    #   - d     : n + m + 1 multiplications
    #   - u3    : n mulitplications
    #   - F11m1 : n**2 multiplications
    mult = n * (m + 2 * n + 2) + m +1

    Graminv = np.vstack(
        (np.hstack((F11m1, np.reshape(-u3, (n, 1)))), np.hstack((-u3, d)))
    )
    perm = np.insert(np.arange(n), idx, -1)
    Graminv = Graminv[perm, :][:, perm]

    return Graminv, mult

def update_reduced_inverse(Gwinv, Xw, X, w, ind):
    """
    Compute the updated Gw^{-1} = (Xw.T @ Xw)^{-1} when the columns of X at 
    index idx are added to Xw. The suffix 'w' means that only the columns 
    selected by the boolean array w are used (ie. Xw = X[:,w], Gw = G[w,w]).
    """

    mult = 0
    ind_Xw = np.where(w)[0]
    for i, ind_i in enumerate(ind):
        find_pos = np.where(ind_Xw >= ind_i)[0]
        if find_pos.size > 0:
            insert_pos = find_pos[0]
        else:
            insert_pos = len(ind_Xw)
        insert_col = X[:,ind_i]
        Gwinv, mult_inv = update_inverse(Gwinv, Xw, insert_pos, insert_col)
        Xw = np.insert(Xw, insert_pos, insert_col, axis=1)
        ind_Xw = np.insert(ind_Xw, 0, ind_i)
        ind_Xw.sort()
        mult += mult_inv

    return Gwinv, mult
