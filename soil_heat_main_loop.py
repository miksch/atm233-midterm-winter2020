import xarray as xr
from constants import *
import soil_funcs
from tdma import *
import matplotlib.pyplot as plt
import numpy as np

for num_depths in [10, 20, 30, 50, 100, 150]:
    num_coeffs = num_depths - 2
    for dz in [.001, .01, .05, .1]:
        depth = dz * num_depths
        for time_steps in [50, 100, 200, 300, 400]:
            for dt in [60, 900, 1800, 3600]:
                la = (dt * kap) / (dz * dz)

                print(la)

                # Initialize temperature, time, and depth arrays
                Temps = np.full((num_depths, time_steps), np.nan)
                tau = np.array([t * dt for t in np.arange(time_steps)])
                depths = np.array([-d * dz for d in np.arange(num_depths)])

                # Initialize boundary conditions in Temps array
                Temps[0, :] = soil_funcs.temp_surface(tau, T_bar, A)  # Surface temperature
                Temps[-1, :] = T_bar  # Temperature at lower boundary
                Temps[:, 0] = T_bar  # Temperature at tau=0

                ## Other options for initializing temperature at tau=0
                # Linear
                # Temps[:, 0] =  np.linspace(T_bar+10, T_bar, num_depths) # Lowest depth = T_bar

                # Diverging from tmax in center
                # gauss = signal.gaussian(n, std=3)
                # Temps[:,0] = gauss

                # Coefficient matrix for tridiagonal solver
                coeffs = np.full((num_coeffs, 4), 0.)
                [a, b, c, d] = [np.zeros(num_coeffs) for _ in np.arange(4)]

                # Calculate temperature array
                Temps = soil_funcs.calc_temps_vector(a, b, c, d, tau, Temps, la, num_coeffs)
                x, y = np.meshgrid(tau, depths)

                da = xr.DataArray(Temps, coords=[('depth', depths), ('tau', tau)]).to_dataset(name='temp')
                da.to_netcdf(f"data/dz_{str(dz).replace('.', '_')}_nd_{num_depths}_dt_{dt}_ts_{time_steps}.nc")

                fig, ax = plt.subplots(**{'figsize' : (10, 5)})
                mesh = ax.pcolormesh(x, y, Temps)
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('Depth [m]')
                fig.suptitle(f"dz: {str(dz).replace('.', '_')}, num_depths: {num_depths}, dt: {dt}, ts: {time_steps}")
                fig.tight_layout()
                fig.savefig(f"figures/dz_{str(dz).replace('.', '_')}_nd_{num_depths}_dt_{dt}_ts_{time_steps}.png", dpi=300)


