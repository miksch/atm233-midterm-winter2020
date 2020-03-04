import xarray as xr
from constants import *
import soil_funcs
from tdma import *
import matplotlib.pyplot as plt
import numpy as np
import os

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

# Create and save data
os.makedirs(f"data", exist_ok=True)
da = xr.DataArray(Temps, coords=[('depth', depths), ('tau', tau)]).to_dataset(name='temp')
da.to_netcdf(f"data/dz_{dz}_nd_{num_depths}_dt_{dt}_ts_{time_steps}_k_{kap:.2e}".replace('.', '_')+".nc")

# Create and save figures
os.makedirs(f"figures/output", exist_ok=True)
fig, ax = plt.subplots(**{'figsize': (10, 5)})
mesh = ax.pcolormesh(x, y, Temps)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Depth [m]')
fig.suptitle(f"dz: {str(dz).replace('.', '_')}, num_depths: {num_depths}, dt: {dt}, ts: {time_steps}, k:{kap:.2e}")
fig.colorbar(mesh, label=r'Temperature [$\degree C$]')
fig.savefig(f"figures/output/dz_{dz}_nd_{num_depths}_dt_{dt}_ts_{time_steps}_k_{kap:.2e}".replace('.', '_')+".png", dpi=300)


