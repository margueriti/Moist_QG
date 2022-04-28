"""
Arrays for an integrating factor method for two layers, possibly with moisture in the bottom layer.

author: Marguerite Brown
email: mlb542@nyu.edu
license: MIT. Please feel free to use and modify this, but keep the above information. Thanks!
"""

from __future__ import division
import numpy as np
import rfft2_spectralmethods2D as sp2
import scipy.linalg as spln

def integratingfactor(param):
    """ The integrating factor to eliminate the stiff portion from timestepping in 2-layer.

    Parameters
    ----------
    param : callable
        Class containing information on the spatial system that f is defined on, details of the QG system, and details of the timestepping.

    Returns
    -------
    ndarray, ndarray
        The matrices for characterizing the stiff portion of the time tendency (L), and the integrating factor exponent (integrating_factor) for 2layer QG.
    """
    dt = param.dt
    layers = param.layers
    print('DT = ',dt)
    L, q2s = stiff_multiplier(param)
    integrating_factor = integratingfactor_loop(L,dt)
    #if param.moisture.any():
    #    integrating_factor2 = integratingfactor_loop(L[:layers,:layers],dt)
    #    integrating_factor[:layers,:layers] = integrating_factor2
    #    integrating_factor[layers:,:layers] = 0.0*integrating_factor[layers:,:layers]
    #    print 'low',np.max(integrating_factor[1,1]-integrating_factor[2,1]-integrating_factor[2,2])
    return integrating_factor,L, q2s

def integratingfactor_loop(L,dt):
    """ The integrating factor to eliminate the stiff portion from timestepping in 2-layer.

    Parameters
    ----------
    L : ndarray
        Matrix characterizing the stiff portion of the time tendency.
    dt : ndarray
        timestep size

    Returns
    -------
    ndarray, ndarray
        The matrices for converting from PV to streamfunction (q2s), and the integrating factor exponent (integrating_factor) for 2layer QG.
    """
    integrating_factor = np.zeros_like(L)
    for i in range(0,len(L[0,0])):
        for j in range(0,len(L[0,0,0])):
            # method described at https://numericalanalysisprogramming.wordpress.com/2017/03/03/matrix-exponential/
            #diag, S = np.linalg.eig(dt*L[:,:,i,j])
            #Sinv = np.linalg.pinv(S)#pseudoinverse
            #D = np.diag(np.exp(diag))
            #S = np.asmatrix(S)
            #D = np.asmatrix(D)
            #Sinv = np.asmatrix(Sinv)
            #integrating_factor[:,:,i,j] = S*D*Sinv
            integrating_factor[:,:,i,j] = spln.expm(dt*L[:,:,i,j])
    return integrating_factor

def q2s_conversion(param):
    """ The multiplier corresponding to the stiff portion of the integrating factor.

    Parameters
    ----------
    param : callable
        Class containing information on the spatial system that f is defined on, details of the QG system, and details of the timestepping.

    Returns
    -------
    ndarray, ndarray
        The matrices for converting from PV to streamfunction (q2s), and the multiplier for the stiff portion of the equation (L) for 2layer QG, with option for additional moist layer.
    """
    nx = param.Nx//2+1#for rfft2
    ell0 = np.ones((param.Ny,nx),dtype=complex)
    laplace=sp2.spectral_Laplacian2(ell0,param)
    ell3 = (laplace-(param.RWN[0]**2+param.RWN[1]**2)*ell0)*laplace
    ell3[0,0]=1.0
    q2s = np.zeros((param.layers,param.layers,len(ell0),len(ell0[0])),dtype=complex)
    for i in range(0,param.layers):
        ell = laplace-param.RWN[i]**2*ell0
        q2s[i,i,:,:] = ell/ell3
        q2s[i,1-i,:,:] = -param.RWN[i]**2/ell3#This is not truly generalized yet. Fix later.
    q2s[:,:,0,0]=np.array([[0.0,0.0],[0.0,0.0]])
    return q2s

def stiff_multiplier(param):
    """ The multiplier corresponding to the stiff portion of the integrating factor.

    Parameters
    ----------
    param : callable
        Class containing information on the spatial system that f is defined on, details of the QG system, and details of the timestepping.

    Returns
    -------
    ndarray, ndarray
        The matrices for converting from PV to streamfunction (q2s), and the multiplier for the stiff portion of the equation (L) for 2layer QG, with option for additional moist layer.
    """
    nx = param.Nx//2+1#for rfft2
    ell0 = np.ones((param.Ny,nx),dtype=complex)
    U = np.array([1.0,-1.0])
    laplace=sp2.spectral_Laplacian2(ell0,param)
    hyper = np.abs(laplace)**4
    hyperm = np.abs(laplace)**4
    ell4 = sp2.spectral_x_deriv(ell0,param)
    ell3 = (laplace-(param.RWN[0]**2+param.RWN[1]**2)*ell0)*laplace
    ell3[0,0]=1.0
    layers = param.layers
    moist_layers = param.total_layers
    L = np.zeros((moist_layers,moist_layers,len(ell0),len(ell0[0])),dtype=complex)
    q2s = q2s_conversion(param)
    for i in range(0,layers):
        mult = -(param.beta+U[i]*param.RWN[i]**2)*ell4-param.damping[i]*laplace #streamfunction coefficients of the ith layer
        L[i,i] = -param.mean_velocity[i]*ell4-param.nu*hyper#pv coefficients of the ith layer
        for j in range(0,layers):
            L[i,j]=L[i,j]+mult*q2s[i,j]
    for i in param.moist_indices:
        multm = -param.gamma*ell4-param.damping[i]*laplace#streamfunction coefficients of the mth layer of moist pv
        m = layers+np.sum(param.moisture[:i])
        L[m,m] = -param.mean_velocity[i]*ell4-param.num*hyperm# moist pv coefficients
        for j in range(0,layers):
            L[m,j]=L[m,j]+multm*q2s[i,j]
    return L, q2s