import numpy as np

## Set of parameters we used with a decent looking output
## (uncomment by taking away triple quotes)

# Very long and annoying run
"""
# Define set up of the parameters of the model
n = 1500 # number of vertical grids (includes top and bottom)
n_coeffs = n-2 # number of coefficients for the tridiag solver
dz = 0.001266667 # vertical grid spacing (in meters)
dt = 1 # time step in seconds
depth = dz * n # the depth of the soil modeled
kap = 8e-7 # soil diffusivity (m2 s-1)
la = (dt*kap)/(dz**2) # la as defined with dt*kappa/dz^2 (unitless)
time_steps = 84600*7 # number of time steps to calculate
T_bar = 20. # Average temperature of bottom layer
A = 10. # Amplitude of sine wave for surface layer
"""

# Large dt
"""
## Set of parameters we used with a decent looking output
## (uncomment by taking away triple quotes)
# Define set up of the parameters of the model
n = 30 # number of vertical grids (includes top and bottom)
n_coeffs = n-2 # number of coefficients for the tridiag solver
dz = 0.05 # vertical grid spacing (in meters)
dt = 3600 # time step in seconds
depth = dz * n # the depth of the soil modeled
kap = 8e-7 # soil diffusivity (m2 s-1)
la = (dt*kap)/(dz**2) # la as defined with dt*kappa/dz^2 (unitless)
time_steps = 200 # number of time steps to calculate
T_bar = 20. # Average temperature of bottom layer
A = 10. # Amplitude of sine wave for surface layer
"""

# What we ended up using
# Define set up of the parameters of the model
num_depths = 1001  # number of vertical grids (includes top and bottom)
num_coeffs = num_depths - 2  # number of coefficients for the tridiag solver
dz = 0.0015  # vertical grid spacing (in meters)
dt = 60  # time step in seconds
depth = dz * num_depths  # the depth of the soil modeled
kap = 8e-7  # soil diffusivity (m2 s-1)
la = (dt * kap) / (dz * dz)  # la as defined with dt*kappa/dz^2 (unitless)
num_days = 8  # Number of days to run the model for
time_steps = int((86400 / dt) * num_days) + 1  # number of time steps to calculate
T_bar = 20.  # Average temperature of bottom layer
A = 10.  # Amplitude of sine wave for surface layer

