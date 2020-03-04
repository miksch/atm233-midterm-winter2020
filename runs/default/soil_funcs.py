import numpy as np
from tdma import *

def calc_temps_array(coeffs, tau, la, temps):

    for i, t in enumerate(tau[1:-1]):

        # Index in temperature array
        Temp_idx = i + 1

        # depth = 1
        coeffs[0, 1] = 1 + 2 * la
        coeffs[0, 2] = - la
        coeffs[0, 3] = temps[1, Temp_idx - 1] + la * temps[0, Temp_idx]

        # depth = bottom
        coeffs[-1, 0] = -la
        coeffs[-1, 1] = 1 + 2 * la
        coeffs[-1, 3] = temps[-2, Temp_idx - 1] + la * temps[-1, Temp_idx]

        # Loop through
        for depth in np.arange(coeffs.shape[0])[1:-1]:
            coeffs[depth, 0] = -la
            coeffs[depth, 1] = 1 + 2 * la
            coeffs[depth, 2] = -la
            coeffs[depth, 3] = temps[depth, Temp_idx - 1]

        temps[1:-1, Temp_idx] = TDMAsolver(coeffs)

    return temps

def calc_temps_vector(a, b, c, d, tau, temps, la, n_coeffs):

    ## 2. Finding the coefficents for a, b, c, d
    for i, t in enumerate(tau[1:-1]):

        # Index in temperature array
        Temp_idx = i + 1

        # depth = 1
        b[0] = 1 + 2 * la
        c[0] = - la
        d[0] = temps[1, Temp_idx - 1] + la * temps[0, Temp_idx]

        # depth = bottom
        a[-1] = -la
        b[-1] = 1 + 2 * la
        d[-1] = temps[-2, Temp_idx - 1] + la * temps[-1, Temp_idx]

        # Loop through
        for depth in np.arange(n_coeffs)[1:-1]:
            a[depth] = -la
            b[depth] = 1 + 2 * la
            c[depth] = -la
            d[depth] = temps[depth, Temp_idx - 1]

        temps[1:-1, Temp_idx] = TDMAsolver(a[1:], b, c[:-1], d)

    return temps

def temp_surface(tau, T_bar, A):
    """
    Calculate surface temperature for a set of times (tao)
    """

    omega = (2 * np.pi) / (86400)
    T = T_bar + A * np.sin(omega * tau)  # + np.pi/2 for time offset

    return T