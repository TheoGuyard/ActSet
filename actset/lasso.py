import numpy as np

def primal_value(x, y, reg, Ax):

    pv = 0.5 * np.linalg.norm(y - Ax) ** 2 + reg * np.sum(x)

    # Operations :
    #   - ||y-Ax||_2^2 = <y-Ax,y-Ax>    : y.size multiplications
    #   - 0.5 * ||y-Ax||_2^2            : 1 multiplication
    #   - reg * sum(x)                  : 1 multiplication
    mult = y.size + 2

    return pv, mult

def dual_value(uf, y):

    dv = 0.5 * (1 - np.linalg.norm(uf - y) ** 2)

    # Operations :
    #   - ||uf-y||_2^2 = <uf-y,uf-y>    : y.size multiplications
    #   - 0.5 * (1-||uf-y||_2^2)        : 1 multiplication
    mult = y.size + 1

    return dv, mult

def primal_scaling(x):
    return np.maximum(x, 0)

def dual_scaling(u, Atu, reg):

    if len(Atu) == 0:
        return 0, 1, 0

    maxval = np.max(Atu)

    if maxval <= reg:
        return u, 1, 0

    scaling = reg / maxval
    uf = scaling * u

    # Operations :
    #   - scaling    : 1 multiplication
    #   - u          : u.size multiplications
    mult = u.size + 1

    return uf, scaling, mult

def dual_gap_value(x, A, y, reg):

    m, n = A.shape
    x = primal_scaling(x)
    Ax = A @ x
    u = y - Ax
    Atu = A.T @ u
    uf, _, mult_ds = dual_scaling(u, Atu, reg)
    pv, mult_pv = primal_value(x, y, reg, Ax)
    dv, mult_dv = dual_value(uf, y)
    gap = pv - dv

    # Operations :
    #   - Ax            : n * m multiplications
    #   - Atu           : n * m multiplications
    #   - uf            : mult_ds multiplications
    #   - pv            : mult_pv multiplications
    #   - dv            : mult_dv multiplications
    mult = 2 * n * m + mult_ds + mult_pv + mult_dv

    return gap, mult
