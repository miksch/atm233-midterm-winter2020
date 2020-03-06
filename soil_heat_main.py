# import xarray as xr # uncomment if you want to save data
from config import *
import soil_funcs
import matplotlib.pyplot as plt
import numpy as np
import os

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

# Linear (if adding pi/2 to start at T_bar+A
# Temps_implicit[:, 0] =  np.linspace(T_bar+A, T_bar, num_depths) # Lowest depth = T_bar

# Analytical initial condition for t = 0
Temps_implicit[:, 0] = soil_funcs.temp_analytical(x[:, 0], y[:, 0], T_bar, A, kap)

# Coefficient vectors for tridiagonal solver
[a, b, c, d] = [np.zeros(num_coeffs) for _ in np.arange(4)]

# Calculate implicit and analytical solution
Temps_implicit = soil_funcs.calc_temps_vector(a, b, c, d, tau, Temps_implicit, la, num_coeffs)
Temps_analytical = soil_funcs.temp_analytical(x, y, T_bar, A, kap)
Temps_diff = Temps_implicit - Temps_analytical

# Uncomment to save data to netcdf files (uses xarray library and only saves implicit solution
# os.makedirs(f"data/single_runs".replace('.', '_'), exist_ok=True)
# da = xr.DataArray(Temps, coords=[('depth', depths), ('tau', tau)]).to_dataset(name='temp')
# da.to_netcdf(f"data/single_runs/dz_{dz}_nd_{num_depths}_dt_{dt}_ts_{time_steps}_k_{kap:.2e}_la_{la:.2f}".replace('.', '_')+".nc")

# Create figures and axes
os.makedirs(f"figures/output/single_runs".replace('.', '_'), exist_ok=True)
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, **{'figsize': (10, 15)})

# Plot out implicit, and analytical solution
mesh_implicit = ax1.pcolormesh(x, y, Temps_implicit)
ax1.set_title('Implicit Solution')
mesh_analytical = ax2.pcolormesh(x, y, Temps_analytical)
ax2.set_title('Analytical Solution')

# Plot out the absolute difference
max_diff = np.nanmax(np.abs(Temps_diff))
mesh_diff = ax3.pcolormesh(x, y, Temps_implicit - Temps_analytical, cmap='coolwarm',
                           vmax=max_diff, vmin=-max_diff) # Centers around 0
ax3.set_title('Implicit - Analytical Solution')

# Loop through subplot formatting
for ax, plot in zip([ax1, ax2, ax3], [mesh_implicit, mesh_analytical, mesh_diff]):
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Depth [m]')
    fig.colorbar(plot, ax=ax, label=r'Temperature [$\degree C$]')

# Comment out if you don't want figure title
fig.suptitle(f"dz: {dz}, Num. Depths: {num_depths}, dt: {dt}, Time Steps: {time_steps}, k: {kap:.2e}, la: {la:.2f}")

fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save figure
fig.savefig(f"figures/output/single_runs/dz_{dz}_nd_{num_depths}_dt_{dt}_ts_{time_steps}_k_{kap:.2e}_la_{la:.2f}".replace('.', '_')+".png", dpi=300)


