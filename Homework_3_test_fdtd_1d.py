'''Test script for Homework 3, Computational Photonics, SS 2020:  FDTD method.
'''


import numpy as np
from finite_difference_time_domain import fdtd_1d, Fdtd1DAnimation
from matplotlib import pyplot as plt
import time 


# dark bluered colormap, registers automatically with matplotlib on import
import bluered_dark

plt.rcParams.update({
        'figure.figsize': (12/2.54, 9/2.54),
        'figure.subplot.bottom': 0.15,
        'figure.subplot.left': 0.165,
        'figure.subplot.right': 0.90,
        'figure.subplot.top': 0.9,
        'axes.grid': False,
        'image.cmap': 'bluered_dark',
})

plt.close('all')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# constants
c = 2.99792458e8 # speed of light [m/s]
mu0 = 4*np.pi*1e-7 # vacuum permeability [Vs/(Am)]
eps0 = 1/(mu0*c**2) # vacuum permittivity [As/(Vm)]
Z0 = np.sqrt(mu0/eps0) # vacuum impedance [Ohm]

# geometry parameters
x_span = 18e-6 # width of computatinal domain [m]
n1 = 1 # refractive index in front of interface
n2 = 2 # refractive index behind interface
x_interface = x_span/4 #postion of dielectric interface

# simulation parameters
dx = 15e-9 # grid spacing [m]
time_span = 60e-15 # duration of simulation [s]

Nx = int(round(x_span/dx)) # number of grid points
x = np.linspace(- x_span / 2, x_span / 2, Nx)
# source parameters
source_frequency = 500e12 # [Hz]
source_position = 0 # [m]
source_pulse_length = 1e-15 # [s]

# %% create permittivity distribution and run simulation %%%%%%%%%%%%%%%%%%%%%%
eps_rel = n1 ** 2 * np.ones(Nx)
# 
eps_rel[x > x_interface] = n2 ** 2



times = []

for _ in range(1):
    a = time.time() 
    Ez, Hy, x, t = fdtd_1d(eps_rel, dx, time_span, source_frequency,
                           source_position, source_pulse_length)
    b = time.time() - a 
    times.append(b)

print("Time elapsed {:.4f} in seconds".format(np.mean(times)))
print("Time elapsed stdev {:.4f} in seconds".format(np.std(times) / np.sqrt(len(times))))


# %% make video %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fps = 25
step = t[-1]/fps/30
x = np.linspace(- x_span / 2, x_span / 2, Nx)
ani = Fdtd1DAnimation(x, t, Ez, Hy, x_interface=x_interface,
                       step=step, fps=fps)
plt.show()

# %% create representative figures of the results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

