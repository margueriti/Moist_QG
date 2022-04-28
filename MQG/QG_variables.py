"""
Functions for the quasigeostrophic equations in one layer in the spectral domain with integrating_factoricit trapezoidal method for handling diffusion.

author: Marguerite Brown
email: mlb542@nyu.edu
license: MIT. Please feel free to use and modify this, but keep the above information. Thanks!
"""

from __future__ import division
import numpy as np
import rfft2_spectralmethods2D as sp2
import scipy.linalg as spln

# potential vorticity and related functions
def potential_vorticity_2L(psi,param):
    """ Potential Vorticity for 2 layers in 2D.

    Parameters
    ----------
    psi : ndarray
        Discrete Fourier transform of the streamfunction in two layers, with index 0 indicating the top layer and index 1 indicating the bottom.
    param : callable
        Class containing information on the spatial system that f is defined on, details of the QG system, and details of the timestepping.

    Returns
    -------
    ndarray
        Discrete fft2 of potential vorticity in two layers with moist pv of the bottom layer.
    """
    q = np.zeros_like(psi)
    baroclinic = psi[0]-psi[1]
    laplace2 = sp2.spectral_Laplacian2(psi[1],param)
    q[0] = sp2.spectral_Laplacian2(psi[0],param)-baroclinic*param.RWN[0]**2
    q[1] = laplace2+baroclinic*param.RWN[1]**2
    for i in np.nonzero(param.moisture)[0]:
        m = param.layers+np.sum(param.moisture[:i])
        q[m] = laplace2+(param.RWN[i]**2*baroclinic+psi[m])/(1-param.latent_heating)
    return q

# velocity and related functions
def velocity_onelayer(psi,param):
    """ Velocity from streamfunction for 1 layer in 2D.

    Parameters
    ----------
    psi : ndarray
        Discrete Fourier transform of the streamfunction in one layer.
    param : callable
        Class containing information on the spatial system that f is defined on, details of the QG system, and details of the timestepping.

    Returns
    -------
    ndarray
        FT of Velocity as (u,v) for 1 layer.
    """
    u = -sp2.spectral_y_deriv(psi,param)
    v = sp2.spectral_x_deriv(psi,param)
    return u,v

def velocity_multilayer(psi,param):
    """ Velocity for n layers in 2D.

    Parameters
    ----------
    psi : ndarray
        Discrete Fourier transform of the streamfunction in n layers.
    param : callable
        Class containing information on the spatial system that f is defined on, details of the QG system, and details of the timestepping.

    Returns
    -------
    ndarray
        Velocity as (u,v) for n layers.
    """
    u = np.zeros_like(psi)
    v = np.zeros_like(u)
    for i in range(0,param.layers):
        u[i],v[i] = velocity_onelayer(psi[i],param)
    return u,v

def velocitymag_multilayer(psi,param):
    """ Velocity magnitude.

    Parameters
    ----------
    psi : ndarray
        Discrete Fourier transform of the streamfunction in n layers.
    param : callable
        Class containing information on the spatial system that f is defined on, details of the QG system, and details of the timestepping.

    Returns
    -------
    ndarray
        Magnitude of velocity for n layers.
    """
    u_hat,v_hat = velocity_multilayer(psi,param)
    u = np.fft.irfft2(u_hat)
    v = np.fft.irfft2(v_hat)
    uconj = np.conjugate(u)
    vconj = np.conjugate(v)
    utot = np.sqrt(u*uconj+v*vconj)
    return utot