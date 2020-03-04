import numpy as np
from numba import jit

# Tridiag solver from Carnahan
def TDMAsolver_carnahan(A, B, C, D):
    # send the vectors a, b, c, d with the coefficents

    vector_len = D.shape[0]  # defines the length of the coefficent vector (including a = 0)
    V = np.zeros(vector_len)  # solution vector
    beta = np.zeros(vector_len)  # temp vector
    gamma = np.zeros(vector_len)  # temp vector

    beta[0] = B[0]
    gamma[0] = D[0] / beta[0]

    for i in range(1, vector_len):
        beta[i] = B[i] - (A[i] * C[i - 1]) / beta[i - 1]
        gamma[i] = (D[i] - A[i] * gamma[i - 1]) / beta[i]

    # compute the final solution vector sv
    V[vector_len - 1] = gamma[vector_len - 1]
    for k in np.flip(np.arange(vector_len - 1)):
        # k = vector_len - i
        V[k] = gamma[k] - C[k] * V[k + 1] / beta[k]

    return V

## Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
# https://gist.github.com/cbellei/8ab3ab8551b8dfc8b081c518ccd9ada9
# Modified to take in coefficient array
def TDMAsolver_no_vec(a, b, c, d):
    """
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    """

    # a = coeffs[1:, 0]
    # b = coeffs[:, 1]
    # c = coeffs[:-1, 2]
    # d = coeffs[:, 3]

    nf = len(d)  # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, 1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    return xc

# https://stackoverflow.com/questions/8733015/tridiagonal-matrix-algorithm-tdma-aka-thomas-algorithm-using-python-with-nump
@jit
def TDMAsolver(a, b, c, d):
    # Set up diagonal coefficients
    n = len(d)
    w = np.zeros(n - 1)
    g = np.zeros(n)
    p = np.zeros(n)

    w[0] = c[0] / b[0]
    g[0] = d[0] / b[0]

    for i in range(1, n - 1):
        w[i] = c[i] / (b[i] - a[i - 1] * w[i - 1])
    for i in range(1, n):
        g[i] = (d[i] - a[i - 1] * g[i - 1]) / (b[i] - a[i - 1] * w[i - 1])
    p[n - 1] = g[n - 1]
    for i in range(n - 1, 0, -1):
        p[i - 1] = g[i - 1] - w[i - 1] * p[i]
    return p
