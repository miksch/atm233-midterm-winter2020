# import xarray as xr # Uncomment if you want to save data
from config import *
import soil_funcs
from tdma import *
import matplotlib.pyplot as plt
import numpy as np
import os

# Looping through certain pairs of depth and dz to have a total depth of 1.5 m
for dz, num_depths in zip([.15, .05, .03, .015, .01, .001],  # dz [m]
                          [11, 31, 51, 101, 151, 1501]  # num_depths (extra 1 to include top layer z=0)
                          ):

    # Calculate total depth and number of coefficients for tdma solver
    num_coeffs = num_depths - 2
    depth = dz * (num_depths - 1)

    # Loop through pairs of time steps and dt to have a total length of 8 days
    for dt, time_steps in zip([60, 900, 1800, 3600],  # dt
                              [11521, 769, 385, 193]  # time_steps (extra 1 to include time = 0)
                              ):

            total_time = dt * (time_steps - 1)
            la = (dt * kap) / (dz * dz)

            # Initialize temperature, time, and depth arrays
            Temps_implicit = np.full((num_depths, time_steps), np.nan)
            tau = np.array([t * dt for t in np.arange(time_steps)])
            depths = np.array([-d * dz for d in np.arange(num_depths)])
            x, y = np.meshgrid(tau, depths)

            # Initialize top and bottom boundary conditions in Temps array
            Temps_implicit[0, :] = soil_funcs.temp_surface(tau, T_bar, A)  # Surface temperature
            Temps_implicit[-1, :] = T_bar  # Temperature at lower boundary

            ## Other options for initializing temperature at tau=0
            # Constant value
            # Temps_implicit[:, 0] = T_bar  # Temperature at tau=0

            # Linear (if adding pi/2 in surface sine wave)
            # Temps_implicit[:, 0] =  np.linspace(T_bar+A, T_bar, num_depths) # Lowest depth = T_bar

            # Analytical initial condition for t = 0
            Temps_implicit[:, 0] = soil_funcs.temp_analytical(x[:, 0], y[:, 0], T_bar, A, kap)

            ## Calculate temperatures using the implicit method
            # Coefficient vectors for tridiagonal solver
            [a, b, c, d] = [np.zeros(num_coeffs) for _ in np.arange(4)]

            # Calculate implicit solution
            Temps_implicit = soil_funcs.calc_temps_vector(a, b, c, d, tau, Temps_implicit, la, num_coeffs)

            # Uncomment to create and save data in netcdf
            # os.makedirs(f"data/{depth}_m/{time_steps*dt}_s".replace('.', '_'), exist_ok=True)
            # da = xr.DataArray(Temps, coords=[('depth', depths), ('tau', tau)]).to_dataset(name='temp')
            # da.to_netcdf(
            #    f"data/{depth}_m/{time_steps*dt}_s/dz_{dz}_nd_{num_depths}_dt_{dt}_ts_{time_steps}_k_{kap:.2e}_la_{la:.2f}".replace(
            #        '.', '_') + ".nc")

            # Create and save figures
            os.makedirs(f"figures/output/{depth:.2f}_m/{total_time}_s/".replace('.', '_'), exist_ok=True)
            fig, ax = plt.subplots(**{'figsize': (10, 5)})

            # Plot data (can change to "ax.contourf(...)" for contour plots)
            mesh = ax.pcolormesh(x, y, Temps_implicit)

            # Formatting and labels
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Depth [m]')
            ax.set_ylim(-depth, 0)
            ax.set_xlim(0, total_time)
            fig.suptitle(f"dz: {dz}, Num. Depths: {num_depths}, dt: {dt}, Time Steps: {time_steps}, k: {kap:.2e}, la: {la:.2f}")
            fig.colorbar(mesh, label=r'Temperature [$\degree C$]')

            # Save figure
            fig.savefig(
                f"figures/output/{depth:.2f}_m/{total_time}_s/dz_{dz}_nd_{num_depths}_dt_{dt}_ts_{time_steps}_k_{kap:.2e}_la_{la:.2f}".replace(
                    '.', '_') + ".png", dpi=300)
