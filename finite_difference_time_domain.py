'''Homework 3, Computational Photonics, SS 2020:  FDTD method.
'''

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import skimage.transform as skf
import time

def fdtd_1d(eps_rel, dx, time_span, source_frequency, source_position,
            source_pulse_length):
    '''Computes the temporal evolution of a pulsed excitation using the
    1D FDTD method. The temporal center of the pulse is placed at a
    simulation time of 3*source_pulse_length. The origin x=0 is in the
    center of the computational domain. All quantities have to be
    specified in SI units.

    Arguments
    ---------
        eps_rel : 1d-array
            Rel. permittivity distribution within the computational domain.
        dx : float
            Spacing of the simulation grid (please ensure dx <= lambda/20).
        time_span : float
            Time span of simulation.
        source_frequency : float
            Frequency of current source.
        source_position : float
            Spatial position of current source.
        source_pulse_length :
            Temporal width of Gaussian envelope of the source.

    Returns
    -------
        Ez : 2d-array
            Z-component of E(x,t) (each row corresponds to one time step)
        Hy : 2d-array
            Y-component of H(x,t) (each row corresponds to one time step)
        x  : 1d-array
            Spatial coordinates of the field output
        t  : 1d-array
            Time of the field output
    '''
    # speed of light [c]=m/s
    c = 2.99792458e8
    # vacuum permeability [mu0]=Vs/(Am)
    mu0 = 4*np.pi*1e-7
    # vacuum permittivity [eps0]=As/(Vm)
    eps0 = 1/(mu0*c**2)
    # vacuum impedance [Z0]=Ohm
    Z0 = np.sqrt(mu0/eps0)

    # choose dt small enough
    dt = dx / 2 / c
    # time array
    t = np.arange(0, time_span, dt)

    # create output x coordinates
    x_width = dx * len(eps_rel)
    x = np.linspace(- x_width / 2, x_width / 2, len(eps_rel))

    # Ez and Hy are shifted half a index step
    # however, we represent them as a simple array
    # we just need to keep in mind that they are shifted by half a index step
    # create output electric Ez
    Ez = np.zeros((len(x), len(t)), dtype=np.complex128)
    # create output magnetic Hy
    Hy = np.zeros((len(x) - 1, len(t)), dtype=np.complex128)

    t0 = 3 * source_pulse_length
    x_index_center = np.argmax(x >= source_position)
    print("xindex, center", x_index_center)
    # function to create jz at different points in time
    def jz_f(n):
        # current time
        t = dt * (n)
        jz = np.zeros(len(x), dtype=np.complex128)
        # set center to source
        jz[x_index_center] = np.exp(-2 * np.pi * 1j * source_frequency * t) *\
                             np.exp(-(t - t0) ** 2 / source_pulse_length ** 2)
        return jz
    # update iterations
    for n in range(1, len(t)):
        jz = jz_f(n - 1)
        # print(jz.shape)
        # print(Ez.shape)
        Ez[1:-1, n] = Ez[1:-1, n - 1] + 1 / (eps_rel[1:-1] * eps0) * dt /\
                      dx * (Hy[1:, n - 1] - Hy[:-1, n - 1]) -\
                      dt / (eps_rel[1:-1] * eps0) * jz[1:-1]
        Hy[:, n] = Hy[:, n - 1] + 1 / mu0 * dt / dx *\
                   (Ez[1:, n]  - Ez[:-1, n])


 #   x_new = np.linspace(- x_width / 2, x_width / 2, 2 * len(eps_rel))

 #   Ez_new = np.zeros((2 * len(x), len(t)), dtype=np.complex128)
 #   Ez_new[::2] = Ez
 #   # simple linear interpolation between the field points
 #   Ez_new[1:-1:2] = (Ez[:-1] + Ez[1:]) / 2
    
    # create output magnetic Hy with boundaries left and right
    Hy_new = np.copy(Hy)#np.zeros((len(x) + 1, len(t)), dtype=np.complex128)
    Hy_new = np.concatenate((np.array([Hy[0, :]]),\
                             Hy[:, :],\
                             np.array([Hy[-1, :]])),\
                            axis=0)

    # create another column for the first time step
    Hy_new2 = np.zeros((len(x) + 1, len(t) + 1), dtype=np.complex128)
    Hy_new2[:, 1:] = Hy_new
    Hy_new2[:, 0] = Hy_new[:, 0]
    Hy_new = Hy_new2
    
    # simple linear 4 point interpolation between the field points
    Hy_new = (Hy_new[:-1, :-1] + Hy_new[1:, :-1] + \
              Hy_new[:-1, 1:] + Hy_new[1:, 1:]) / 4

    Ez = np.swapaxes(Ez, 0, 1)
    Hy_new = np.swapaxes(Hy_new, 1, 0)

    return Ez, Hy_new, x, t


def fdtd_3d(eps_rel, dr, time_span, freq, tau, jx, jy, jz,
            field_component, z_ind, output_step):
    '''Computes the temporal evolution of a pulsed spatially extended current
    source using the 3D FDTD method. Returns z-slices of the selected
    field at the given z-position every output_step time steps. The pulse
    is centered at a simulation time of 3*tau. All quantities have to be
    specified in SI units.

    Arguments
    ---------
        eps_rel: 3d-array
            Rel. permittivity distribution within the computational domain.
        dr: float
            Grid spacing (please ensure dr<=lambda/20).
        time_span: float
            Time span of simulation.
        freq: float
            Center frequency of the current source.
        tau: float
            Temporal width of Gaussian envelope of the source.
        jx, jy, jz: 3d-array
            Spatial density profile of the current source.
        field_component : str
            Field component which is stored (one of ‘ex’,’ey’,’ez’,
            ’hx’,’hy’,’hz’).
        z_index: int
            Z-position of the field output.
        output_step: int
            Number of time steps between field outputs.

    Returns
    -------
        F: 3d-array
            Z-slices of the selected field component at the
            z-position specified by z_ind stored every output_step
            time steps (time varies along the first axis).
        t: 1d-array
            Time of the field output.
    '''
    pass


class Fdtd1DAnimation(animation.TimedAnimation):
    '''Animation of the 1D FDTD fields.

    Based on https://matplotlib.org/examples/animation/subplots.html

    Arguments
    ---------
    x : 1d-array
        Spatial coordinates
    t : 1d-array
        Time
    x_interface : float
        Position of the interface (default: None)
    step : float
        Time step between frames (default: 2e-15/25)
    fps : int
        Frames per second (default: 25)
    Ez: 2d-array
        Ez field to animate (each row corresponds to one time step)
    Hy: 2d-array
        Hy field to animate (each row corresponds to one time step)
    '''

    def __init__(self, x, t, Ez, Hy, x_interface=None, step=2e-15/25, fps=25):
        # constants
        c = 2.99792458e8 # speed of light [m/s]
        mu0 = 4*np.pi*1e-7 # vacuum permeability [Vs/(Am)]
        eps0 = 1/(mu0*c**2) # vacuum permittivity [As/(Vm)]
        Z0 = np.sqrt(mu0/eps0) # vacuum impedance [Ohm]
        self.Ez = Ez
        self.Z0Hy = Z0*Hy
        self.x = x
        self.ct = c*t

        # index step between consecutive frames
        self.frame_step = int(round(step/(t[1] - t[0])))

        # set up initial plot
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        vmax = max(np.max(np.abs(Ez)),np.max(np.abs(Hy))*Z0)*1e6
        fig, ax = plt.subplots(2,1, sharex=True, gridspec_kw={'hspace': 0.4})
        self.line_E, = ax[0].plot(x*1e6, self.E_at_step(0),
                         color=colors[0], label='$\\Re\\{E_z\\}$')
        self.line_H, = ax[1].plot(x*1e6, self.H_at_step(0),
                         color=colors[1], label='$Z_0\\Re\\{H_y\\}$')
        if x_interface is not None:
            for a in ax:
                a.axvline(x_interface*1e6, ls='--', color='k')
        for a in ax:
            a.set_xlim(x[[0,-1]]*1e6)
            a.set_ylim(np.array([-1.1, 1.1])*vmax)
        ax[0].set_ylabel('$\\Re\\{E_z\\}$ [µV/m]')
        ax[1].set_ylabel('$Z_0\\Re\\{H_y\\}$ [µV/m]')
        self.text_E = ax[0].set_title('')
        self.text_H = ax[1].set_title('')
        ax[1].set_xlabel('$x$ [µm]')
        super().__init__(fig, interval=1000/fps, blit=False)

    def E_at_step(self, n):
        return self.Ez[n,:].real*1e6

    def H_at_step(self, n):
        return self.Z0Hy[n,:].real*1e6

    def new_frame_seq(self):
        return iter(range(0, self.ct.size, self.frame_step))

    def _init_draw(self):
        self.line_E.set_ydata(self.x*np.nan)
        self.line_H.set_ydata(self.x*np.nan)
        self.text_E.set_text('')
        self.text_E.set_text('')

    def _draw_frame(self, framedata):
        i = framedata
        self.line_E.set_ydata(self.E_at_step(i))
        self.line_H.set_ydata(self.H_at_step(i))
        self.text_E.set_text(
                'Electric field, $ct = {0:1.2f}$µm'.format(self.ct[i]*1e6))
        self.text_H.set_text(
                'Magnetic field, $ct = {0:1.2f}$µm'.format(self.ct[i]*1e6))
        self._drawn_artists = [self.line_E, self.line_H,
                               self.text_E, self.text_H]


class Fdtd3DAnimation(animation.TimedAnimation):
    '''Animation of a 3D FDTD field.

    Based on https://matplotlib.org/examples/animation/subplots.html

    Arguments
    ---------
    x, y : 1d-array
        Coordinate axes.
    t : 1d-array
        Time
    field: 3d-array
        Slices of the field to animate (the time axis is assumed to be be
        the first axis of the array)
    titlestr : str
        Plot title.
    cb_label : str
        Colrbar label.
    rel_color_range: float
        Range of the colormap relative to the full scale of the field magnitude.
    fps : int
        Frames per second (default: 25)
    '''

    def __init__(self, x, y, t, field, titlestr, cb_label, rel_color_range, fps=25):
        # constants
        c = 2.99792458e8 # speed of light [m/s]
        self.ct = c*t

        self.fig = plt.figure()
        self.F = field
        color_range = rel_color_range*np.max(np.abs(field))
        phw = 0.5*(x[1] - x[0]) # pixel half-width
        extent = ((x[0] - phw)*1e6, (x[-1] + phw)*1e6,
                  (y[-1] + phw)*1e6, (y[0] - phw)*1e6)
        self.mapable = plt.imshow(self.F[0,:,:].real.T,
                                  vmin=-color_range, vmax=color_range,
                                  extent=extent)
        cb = plt.colorbar(self.mapable)
        plt.gca().invert_yaxis()
        self.titlestr = titlestr
        self.text = plt.title('')
        plt.xlabel('x position [µm]')
        plt.ylabel('y position [µm]')
        cb.set_label(cb_label)
        super().__init__(self.fig, interval=1000/fps, blit=False)

    def new_frame_seq(self):
        return iter(range(self.ct.size))

    def _init_draw(self):
        self.mapable.set_array(np.nan*self.F[0, :, :].real.T)
        self.text.set_text('')

    def _draw_frame(self, framedata):
        i = framedata
        self.mapable.set_array(self.F[i, :, :].real.T)
        self.text.set_text(self.titlestr
                           + ', $ct$ = {0:1.2f}µm'.format(self.ct[i]*1e6))
        self._drawn_artists = [self.mapable, self.text]


class Timer(object):
    '''Tic-toc timer.
    '''
    def __init__(self):
        '''Initializer.
        Stores the current time.
        '''
        self._tic = time.time()

    def tic(self):
        '''Stores the current time.
        '''
        self._tic = time.time()

    def toc(self):
        '''Returns the time in seconds that has elapsed since the last call
        to tic().
        '''
        return time.time() - self._tic
