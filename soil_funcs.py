import numpy as np
from tdma import *

def calc_temps_vector(a, b, c, d, tau, temps, la, n_coeffs):
    """
    Calculate the coefficient array for the TDMA solver and calculate temperature

    :param a: a vector of length n_coeffs (empty)
    :param b: b vector of length n_coeffs
    :param c: c vector of length n_coeffs
    :param d: d vector of length n_coeffs
    :param tau: time array
    :param temps: temperature array of dimensions (num_depths x num_times)
    :param la: lambda from paper
    :param n_coeffs: number of coefficients (depths - 2)
    :return: temps
    """
    ## 2. Finding the coefficents for a, b, c, d for each timestep
    for i, t in enumerate(tau[1:-1]):
        Temp_idx = i + 1

        # Loop through coefficients
        for depth in np.arange(n_coeffs):
            a[depth] = -la
            b[depth] = 1 + 2 * la
            c[depth] = -la
            d[depth] = temps[depth+1, Temp_idx-1]

        d[0] += la * temps[0, Temp_idx]
        d[-1] += la * temps[-1, Temp_idx]

        # Solve for temperature
        temps[1:-1, Temp_idx] = TDMAsolver(a[1:], b, c[:-1], d)

    return temps

def temp_surface(tau, T_bar, A):
    """
    Calculate temperature at soil surface (z=0)

    :param tau: Times in seconds
    :param T_bar: Scalar value of mean temperature
    :param A: Amplitude of sine array
    :return: 1D array of temperatures
    """

    omega = (2 * np.pi) / (86400)
    T = T_bar + A * np.sin(omega * tau)  # + np.pi/2 for time offset

    return T

def temp_analytical(tau_x, depth_y, T_bar, A, kap):
    """
    Calculate analytical solution for certain times and depth in a 2D array

    :param tau: 2D array of times (time on axis=1)
    :param depths: 2D array of depths (depth on axis=0)
    :param T_bar: Scalar value of the mean temperature
    :param A: Amplitude of temperature sine wave
    :param kap: Soil
    :return: 2D array of temperatures
    """
    p = 24
    w = (2 * np.pi) / (p * 3600)
    D = np.sqrt((2 * kap) / w)

    # correct ts and z_prof to make proper 2d graph
    temps = np.zeros(tau_x.shape)

    # calculate profile
    temps = T_bar + A * np.exp(depth_y / D) * np.sin(w * tau_x + depth_y / D)

    return temps
