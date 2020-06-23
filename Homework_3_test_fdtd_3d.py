'''Test script for Homework 3, Computational Photonics, SS 2020:  FDTD method.
'''


import numpy as np
from finite_difference_time_domain import fdtd_3d, Fdtd3DAnimation
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

# simulation parameters
Nx = 199 # number of grid points in x-direction
Ny = 201 # number of grid points in y-direction
Nz = 5   # number of grid points in z-direction
dr = 30e-9 # grid spacing in [m]
time_span = 10e-15 # duration of simulation [s]

# x coordinates
x = np.arange(-int(np.ceil((Nx-1)/2)), int(np.floor((Nx-1)/2)) + 1)*dr
# y coordinates
y = np.arange(-int(np.ceil((Ny-1)/2)), int(np.floor((Ny-1)/2)) + 1)*dr

# source parameters
freq = 500e12 # pulse [Hz]
tau = 1e-15 # pulse width [s]
t0 = 3 * tau
source_width = 2 * dr # width of Gaussian current dist. [grid points]

# grid midpoints
midx = int(np.ceil((Nx-1)/2))
midy = int(np.ceil((Ny-1)/2))
midz = int(np.ceil((Nz-1)/2))

# choose dt small enough
dt = dr / 2 / c
# time array
t = np.arange(0, time_span, dt)

# %% create relative permittivity distribution %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

eps_rel = np.ones((Nx, Ny, Nz))

# %% current distributions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# jx = jy = np.zeros(...)
# jz : Gaussion distribution in the xy-plane with a width of 2 grid points,
# constant along z



jx = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
jy = np.copy(jx)
jz = np.ones((1, 1, Nz)) * np.exp(- (np.reshape(y, (1, len(y), 1)) ** 2 + 
               np.reshape(x, (len(x), 1, 1)) ** 2) / source_width ** 2)
# jz = np.exp(- (np.reshape(x, (1, len(x))) ** 2 + 
#                np.reshape(y, (len(y), 1)) ** 2) / source_width ** 2)

# output parameters
z_ind = midz  # z-index of field output
output_step = 4  # time steps between field output

#%% run simulations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

times = []
for _ in range(10):
    a = time.time() 
    Ex, Ey, Ez, Hx, Hy, Hz, t =\
        fdtd_3d(eps_rel, dr, time_span, freq, tau, jx, jy, jz, "ex",
                   z_ind, output_step)
    b = time.time() - a 
    times.append(b)

print("Time elapsed {:.4f} in seconds".format(np.mean(times)))
print("Time elapsed stdev {:.4f} in seconds".format(np.std(times) / np.sqrt(len(times))))


x = x[1:]
y = y[1:]
#%% movie of Hx %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F = Hz*Z0*1e6
titlestr = 'x-Component of Magnetic Field'
cb_label = '$\\Re\\{Z_0H_x\\}$ [µV/m]'
rel_color_range = 1/3
fps = 10




ani = Fdtd3DAnimation(x, y, t, F, titlestr, cb_label, rel_color_range, fps)
plt.show()

##%% movie of Ez %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F = Hy*1e6
titlestr = 'z-Component of Electric Field'
cb_label = '$\\Re\\{E_z\\}$ [µV/m]'
rel_color_range = 1/3
fps = 10



ani = Fdtd3DAnimation(x, y, t, F, titlestr, cb_label, rel_color_range, fps)
plt.show()
