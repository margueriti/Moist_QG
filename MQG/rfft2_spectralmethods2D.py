"""
Spectral Methods: Derivatives, Laplacian, and Jacobian in two dimensions using rfft2.

author: Marguerite Brown
email: mlb542@nyu.edu
license: MIT. Please feel free to use and modify this, but keep the above information. Thanks!
"""

from __future__ import division
import numpy as np

def spectral_y_deriv(f,spatial,n=1):
    """ Take the nth y derivative of a function in 2D.

    Parameters
    ----------
    f : ndarray
        The rfft2 of a function truncated to n x m elements.
    spatial : callable
        Class containing information on the spatial system that f is defined on.
    n : int, optional
        The order of the derivative (default = 1)

    Returns
    -------
    ndarray
        The nth y derivative of a function in 2D.
    """
    fy = np.zeros_like(f)
    for i in range(0,len(f)):
        fy[i,:] = (1j*spatial.freqsy[i])**n*f[i,:]
    return fy

def spectral_x_deriv(f,spatial,n=1):
    """ Take the nth x derivative of a function in 2D.

    Parameters
    ----------
    f : ndarray
        The rfft2 of a function truncated to n x m elements.
    spatial : callable
        Class containing information on the spatial system that f is defined on.
    n : int, optional
        The order of the derivative (default = 1)

    Returns
    -------
    ndarray
        The nth x derivative of a function in 2D
    """
    fx = np.zeros_like(f)
    for i in range(0,len(f[0])):
        fx[:,i] = (1j*spatial.freqsx[i])**n*f[:,i]
    return fx

def spectral_Laplacian2(f,spatial):
    """ Take the Laplacian in Fourier space in 2D.

    Parameters
    ----------
    f : ndarray
        The rfft2 of a function truncated to n x m elements.
    spatial : callable
        Class containing information on the spatial system that f is defined on.

    Returns
    -------
    ndarray
        The rfft2 of the Laplacian of the initial function truncated to n x m elements.
    """
    u = spectral_x_deriv(f,spatial,2)+spectral_y_deriv(f,spatial,2)
    return u

def spectral_Jacobian(f,g,spatial):
    """ Take the Jacobian of two functions in 2D.

    Parameters
    ----------
    f : ndarray
        The rfft2 of a function truncated to n x m elements.
    g : ndarray
        The rfft2 of a second function truncated to n x m elements.
    spatial : callable
        Class containing information on the spatial system that f is defined on.

    Returns
    -------
    ndarray
        The discrete Fourier transform of the Laplacian of the initial function truncated to n x m elements.
    """
    fx = np.fft.irfft2(spectral_x_deriv(f,spatial))
    fy = np.fft.irfft2(spectral_y_deriv(f,spatial))
    gx = np.fft.irfft2(spectral_x_deriv(g,spatial))
    gy = np.fft.irfft2(spectral_y_deriv(g,spatial))
    J = np.fft.rfft2(fx*gy-fy*gx)
    return J

def spectral_Jacobian_antialiasing(f,g,spatial):
    """ Take the Jacobian of two functions in 2D.

    Parameters
    ----------
    f : ndarray
        The rfft2 of a function truncated to n x m elements.
    g : ndarray
        The rfft2 of a second function truncated to n x m elements.
    spatial : callable
        Class containing information on the spatial system that f is defined on.

    Returns
    -------
    ndarray
        The discrete Fourier transform of the Laplacian of the initial function truncated to n x m elements.
    """
    fx = padding(spectral_x_deriv(f,spatial),spatial)
    fy = padding(spectral_y_deriv(f,spatial),spatial)
    gx = padding(spectral_x_deriv(g,spatial),spatial)
    gy = padding(spectral_y_deriv(g,spatial),spatial)
    J_pad = np.fft.rfft2(fx*gy-fy*gx)
    J = aa_truncation(J_pad,spatial)
    return J

def double_spectral_Jacobian_antialiasing(f,g,h,spatial):
    """ Take the Jacobian of two functions in 2D.

    Parameters
    ----------
    f : ndarray
        The rfft2 of a streamfunction truncated to n x m elements.
    g : ndarray
        The rfft2 of a dry pv function truncated to n x m elements.
    h : ndarray
        The rfft2 of a moist pv function truncated to n x m elements.
    spatial : callable
        Class containing information on the spatial system that f is defined on.

    Returns
    -------
    ndarray
        The discrete Fourier transform of the Laplacian of the initial function truncated to n x m elements.
    """
    fx = padding(spectral_x_deriv(f,spatial),spatial)
    fy = padding(spectral_y_deriv(f,spatial),spatial)
    gx = padding(spectral_x_deriv(g,spatial),spatial)
    gy = padding(spectral_y_deriv(g,spatial),spatial)
    hx = padding(spectral_x_deriv(h,spatial),spatial)
    hy = padding(spectral_y_deriv(h,spatial),spatial)
    J_pad = np.fft.rfft2(fx*gy-fy*gx)
    J1 = aa_truncation(J_pad,spatial)
    J_pad = np.fft.rfft2(fx*hy-fy*hx)
    J2 = aa_truncation(J_pad,spatial)
    return J1,J2

def symmetrize_rfft2(f):
    """ Symmetrizes the highest frequency mode of an rfft2 with even number of modes.

    Parameters
    ----------
    f : ndarray
        Fourier transform of a function.

    Returns
    -------
    ndarray
        Fourier transform of a function with an odd number of elements.
    """
    n=len(f)
    if n%2==0:
        mid = n//2
        unmatched = f[-mid,:]/2
        f = np.append(np.append(f[:mid,:],[unmatched,unmatched],axis=0),f[-mid+1:,:],axis=0)*(n+1)/n
    return f

def interpft2(f,N,M):
    """ Take a truncated rfft2 of length (n,m)<(N,M). Interpolate to (N,M) points.

    Parameters
    ----------
    f : ndarray
        Discrete Fourier transform truncated to n<N elements.
    N : int
        The number of points output in x direction.
    M : int
        The number of points output in y direction.

    Returns
    -------
    ndarray
        Discrete rfft2 of a function truncated to (N,M) elements.
    """
    f = symmetrize_rfft2(f)
    m = len(f)
    n = len(f[0])
    mid = m//2
    n0 = 2*(n-1)#assuming n0 even
    N2 = N//2+1
    pad = (M/m)*np.append(np.append(f[:-mid,:],np.zeros((M-m,n)),axis=0),f[-mid:,:],axis=0)
    pad = (N/n0)*np.append(pad,np.zeros((M,N2-n)),axis=1)
    return pad

def aa_truncation(w_pad_hat,spatial):
    what = spatial.renorm*np.append(w_pad_hat[0:spatial.mid2,:spatial.Nk],w_pad_hat[-spatial.mid:,:spatial.Nk],axis=0) #truncate to original length
    return what

def padding(uhat,spatial):
    #note: this will be off by a normalization factor if left as is, but is more efficient for the purposes of reversible anti-aliasing.
    uhat_pad=np.append(uhat, 1j*np.zeros((spatial.Nl,spatial.M)),axis=1)
    uhat_pad=np.append(np.append(uhat_pad[0:spatial.mid2,:], 1j*np.zeros((spatial.N,spatial.Nk2)),axis=0),uhat_pad[-spatial.mid:,:],axis=0)
    u_pad = np.fft.irfft2(uhat_pad)
    return u_pad

def antialias2(uhat,vhat,spatial):
    """ Anti-aliasing the product of two truncated Fourier transforms in 2D. Number of points must be mod 4.

    Parameters
    ----------
    uhat : ndarray
        The discrete rfft2 of a function truncated to nxm elements.
    vhat : ndarray
        The discrete rfft2 of a function truncated to nxm elements.

    Returns
    -------
    ndarray
        The Fourier transform of the product of the two inputs, truncated to nxm elements.
    """
    u_pad = padding(uhat,spatial) #returning to original function with interpolated data
    v_pad = padding(vhat,spatial) #returning to original function with interpolated data
    w_pad = u_pad*v_pad #product in real space
    w_pad_hat = np.fft.rfft2(w_pad) #return to Fourier space
    what = aa_truncation(w_pad_hat,spatial) #truncate to original length
    return what

def parseval2(uhat,vhat):
    """ Parseval's identity for two arrays of rfft2s in time, of shape (time, n,m).

    Parameters
    ----------
    uhat : ndarray
        The rfft2 of a function in time with shape (time, n, m).
    vhat : ndarray
        The rfft2 of a function in time with shape (time, n, m).

    Returns
    -------
    ndarray
        The mean value of the product of the two functions by Parseval's theorem in time, of shape (time).
    """
    n = len(uhat[0,:,0])
    m = (len(uhat[0,0,:])-1)*2
    prod = np.real(np.conj(uhat)*vhat)
    if m % 2:  # odd-length
        mean = (prod[:,:,0] + 2*np.sum(prod[:,:,1:],axis=2))/m
    else:  # even-length
        mean = (prod[:,:,0] + 2*np.sum(prod[:,:,1:-1],axis=2) + prod[:,:,-1])/m
    mean2 = np.sum(mean, axis=1)/n
    return mean2/n/m
