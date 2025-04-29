"""
Contains miscellaneous operations and convenience functions.

Contact: Violet Player
Email: violet.player@noaa.gov
"""

#===================================== Imports =====================================#

import numpy as np
from forecast import states

#===================================== Array Operations =====================================#

def interpolate_matrix(x : states._array_like,
                       xp : states._array_like, 
                       A : np.ndarray, 
                       method="constant", 
                       axis=-1) -> np.ndarray:
    """
    Interpolates a 2D matrix along the requested axis

    Arguments:
        x : new axis of interpolation
        xp : original axis (same units as x)
        A : 2D array
        method : str, [constant, linear] interpolation
        axis : which of two axes to interpolate along?
    """

    ### array-ify
    x = np.array(x)
    xp = np.array(xp)

    ### build new matrix shape
    dim0, dim1 = A.shape
    if axis == 0:
        dim0 = len(x)
    else:
        dim1 = len(x)
    A_itp = np.zeros((dim0, dim1))

    #### iterate over other axis to interpolate
    for i in range(A_itp.shape[np.abs(axis) - 1]):
        if axis == 0:
            A_itp[:,i] = _interp_1D(x, xp, A[:,i], method)
        else:
            A_itp[i,:] = _interp_1D(x, xp, A[i,:], method)

    return A_itp

def _interp_1D(x, xp, yp, method="constant"):

    if method.lower() == "constant":
        y_itp = np.zeros(len(x))
        for i in range(len(x)):
            y_itp[i] = yp[np.argmin(np.abs(xp - x[i]))]

    elif method.lower() == "linear":
        y_itp = np.interp(x, xp, yp)

    else:
        raise Exception("Choose a valid interpolation method: [constant, linear]")
    
    return y_itp

#===================================== Math =====================================#

def RKO4_step(func : states.FunctionType, 
               y0 : states.Union[float, np.ndarray],
               t0 : float, 
               h : float, 
               *args) -> states.Union[float, np.ndarray]:
    """
    Takes a single step in a Runge-Kutta O(4) solver for equations of the form dy/dt = f(t,y).
    Makes four function calls per time step.

    Presumes you will do a good job keeping function output shapes consistent, do pay attention to that.

    Arguments:
        func : f(t,y) in the equation.
        y0 : Solution on previous step (or initial condition)
        t0 : Time at which y(t) = y0
        h : Time step (same units as t0)
        *args : any additional arguments to be passed to func
    
    Returns:
        Solution for y at time t0 + h
    """
    k1 = func(t0, y0, *args)
    k2 = func(t0 + h/2.0, y0 + k1*h/2.0, *args)
    k3 = func(t0 + h/2.0, y0 + k2*h/2.0, *args)
    k4 = func(t0 + h, y0 + k3*h, *args)
    return y0 + (h/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)