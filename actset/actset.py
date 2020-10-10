import time
import numpy as np
from actset.lasso import dual_gap_value, primal_value


def actset(x0, A, y, reg, prec=1e-10, maxit=int(1e4), maxop=int(1e9)):
    """
    Solves the non-negative LASSO problem
        argmin_(x>=0) { 0.5 * ||y-Ax||^2_2 + reg * ||x||_1 }
    using an Active-set method.

    Parameters
    ----------
    x0: (n,) np.ndarray
        Initial point.
    A: (m,n) np.ndarray
        Dictionary of the problem.
    y: (m,) np.ndarray
        Data vector of the problem.
    reg: float (unique regularization) or np.array (continuation sequence)
        Regularization amount between 0 and 1. If reg is a np.array,
        continuation method is used.
    prec: float
        Target gap precision.
    maxit: int
        Maximum number of iterations (for each regularization if continuation 
        is enabled).
    maxop: int
        Maximum number of multiplications (for each regularization if 
        continuation is enabled).

    Returns
    -------
    x: np.array
        Solution of the problem.
    monitor: dict
        Monitoring values.

    Raises
    ------
    ValueError
        If parameters are not well specified.

    Example
    ------
    >>> import numpy as np
    >>> from turbo_screening.actset import actset
    >>> from turbo_screening.dictionary import sample_data
    >>> x0, A, y = np.zeros(60), np.random.rand(50,60), np.random.rand(50)
    >>> reg = 0.3
    >>> x, monitor = actset(x0, A, y, reg, prec=1e-5)
    """

    if A.shape[1] != x0.size or A.shape[0] != y.size:
        raise ValueError("x0, A, y must be (n,), (m,n) and (m,) numpy arrays")

    if np.any(reg < 0):
        raise ValueError("Positive value(s) expected for parameter reg")

    if type(prec) is not float:
        raise ValueError("Expected float for parameter prec")

    if prec <= 0 :
        raise ValueError("Expected a stricly positive value for parameter prec")
    
    if type(maxit) is not int:
        raise ValueError("Expected int for parameter maxit")
    
    if type(maxop) is not int:
        raise ValueError("Expected int for parameter maxop")

    if np.array(reg).shape:
        reg = np.sort(reg)[::-1]
        return _actset_continuation(x0, A, y, reg, param)
    else:
        return _actset(x0, A, y, reg, maxit, prec, maxop)


def _actset(x0, A, y, reg, maxit, prec, maxop):
    """
    Active set algorithm implementation.
    """

    # The linear system of part 1. is solved using the matrix Qinv. This matrice
    # is computed with the first working set and is updated with rank-one rules.
    # To make the code more readable, rank-one rules are not used but 
    # their complexity is taken into account.

    ### Monitoring values ###

    obj = np.full(maxit, np.nan)
    gap = np.full(maxit, np.nan)
    mult = np.zeros(maxit)
    w_size = np.zeros(maxit)
    conv = 'No convergence'


    ### Initialization ###

    m, n = A.shape
    x = x0.copy()
    w = np.abs(x) < 1e-15           # Working set
    wc = np.invert(w)               # Working set complementary
    nwc = np.sum(wc)                # Current number of coefficients in wc
    Awc = A[:, wc]                  # Matrix resticted to the complementary of w       
    Q = Awc.T @ Awc                 # Restricted quadratic term
    Qinv = np.linalg.inv(Q)         # Inverse used in step 1.
    c = np.full(n, reg) - A.T @ y   # Linear term
    
    # Operation count 
    #   Q     : m * nwc^2
    #   Qinv  : nwc * (nwc + 1) * (nwc - 1) / 3
    #   c     : m * n
    mult[0] += m * nwc + m * n + \
        nwc * (nwc + 1) * (nwc - 1) / 3
    w_size[0] = np.sum(w)

    for it in range(maxit):

        ### Active set step ###

        # 1. Solve que QP with forced equality for coefficients in w
        g = A.T @ (A @ x) + c
        p = np.zeros(n)
        p[wc] = - Qinv @ g[wc]

        # Operation count 
        #   g     : 2 * m * n
        #   p     : nwc^2
        mult[it] += 2 * m * n + nwc**2

        # 2.a : Check optimality conditions with Lagrange multipliers
        if np.allclose(p, 0):
            # Lagrange multipliers
            lambd = np.zeros(n)
            lambd[w] = g[w]

            # 2.a.1 : Solution is optimal for the current working set
            if np.all(lambd[w] >= 1e-10):
                # Only for monitoring, no computational cost
                obj[it], _ = primal_value(x, y, reg, A @ x)
                gap[it], _ = dual_gap_value(x, A, y, reg)
                w_size[it] = np.sum(w)
                conv = 'Optimal'
                break
            # 2.a.2 : The current working set is wrong
            else:
                jw = np.argmin(lambd[w])
                j = np.arange(n)[w][jw]
                w[j] = False
                wc = np.invert(w)
                nwc = np.sum(wc)
                Awc = A[:, wc]
                Qinv = np.linalg.inv(Awc.T @ Awc)

                # Operation count 
                #   Qinv     : 2 * nwc^2 + nwc * m + 2 * nwc + m + 1
                mult[it] += 2 * nwc**2 + nwc * m + 2 * nwc + m + 1

        # 2.b : We need to decide how much we move in the direction of p
        else:
            # Step length
            cond = np.logical_and(np.invert(w), p < 0)
            # 2.b.1 : There may be some blocking constraints
            if np.any(cond):
                jcond = np.argmin(-x[cond]/p[cond])
                j = np.arange(n)[cond][jcond]
                alpha = min(1, -x[j]/p[j])

                # Operation count
                #    alpha  : card(cond)
                mult[it] += np.sum(cond)

                # There are some blocking constraints
                if alpha < 1:
                    w[j] = True
                    wc = np.invert(w)
                    nwc = np.sum(wc)
                    Awc = A[:, wc]
                    Qinv = np.linalg.inv(Awc.T @ Awc)

                    # Operation count
                    #   Qinv    : nwc * (nwc + 1)
                    mult[it] += nwc * (nwc + 1)

            # 2.b.2 : No blocking constraints
            else:
                alpha = 1
            # Move toward p
            x += alpha * p

            # Operation count
            #   x   : nwc (only non null coefficients are multiplied)
            mult[it] += nwc 

        # Only for monitoring, no computational cost
        obj[it], _ = primal_value(x, y, reg, A @ x)
        gap[it], _ = dual_gap_value(x, A, y, reg)
        w_size[it] = np.sum(w)

        # Operation and optimal active set stopping criteria
        if np.sum(mult) > maxop:
            conv = 'Maxop reached'
            break
    
    if it + 1 >= maxit:
        conv = 'Maxit reached'

    monitor = {
        'obj': obj[:it+1],
        'gap': gap[:it+1],
        'mult': mult[:it+1],
        'w_size': w_size[:it+1],
        'conv': conv
    }

    return x, monitor

def _actset_continuation(x0, A, y, reg, param):
    raise NotImplementedError