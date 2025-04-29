#### generic python
import numpy as np
import fnmatch
from itertools import product

#### module imports
import models.states as states
import models.utils as utils

def SINDYc_forecast(job : states.TaskState):
    """
    Method for the SINDYc forecast

    Arguments:
        job : JobState variable, contains necessary runtime data.

    Returns:
        fcst : The forecast - solution to the SINDY ODE for the given inputs.
    """

    #### set up inputs
    state = job.ModelState # alias
    t_rkf = job.parameters["solver_times_full"] # Runge-Kutta full-step times
    t_rkh = job.parameters["solver_times_half"] # RK half-step times
    t_drive = job.parameters["driver_times"] # driver time index
    drivers = job.parameters["drivers"]
    drivers_itp = _interpolate_drivers_SINDYc(t_rkh, t_drive, drivers, state) # interpolate to half-step times

    #### set up RKO4 solver and solution container
    fcst = np.zeros((len(t_rkf), job.initial_state.size))
    fcst[0,:] = job.initial_state
    
    #### Forecast!
    for i in range(1,len(t_rkf)):
        Xi = _get_coeffs_SINDYc(drivers[i,state.indices["kp"]], state)
        fcst[i,:] = utils.RKO4_step(_ODE_SINDYc, 
                                    fcst[i-1,:].reshape((-1,1)),
                                    t_rkf[i],
                                    state.dt,
                                    *[t_rkh, drivers_itp, Xi, state]).squeeze()
        
    if job.parameters["forecast_times"] is not None:
        fcst = utils.interpolate_matrix(job.parameters["forecast_times"], t_rkf, 
                                        fcst, state.interp_method, axis=0)

    return fcst

def construct_basis_SINDYc(q : states._array_like, 
                           u : states._array_like,
                           basis_library : list[str] = ["linear"],
                           custom_library : dict = None, 
                           *args) -> tuple[list[states.FunctionType], list[str]]:
    """
    Method that constructs 

    Arguments:
        q : Array-like, represents feature space. Doesn't need to hold actual data.
        u : Array-like, represents driver space. Doesn't need to hold actual data.
        basis_library : list[str], holds list of basis library function types.
        custom_library : Dictionary with keys ["functions", "names"] holding the custom functions. custom_library["functions"] should be a list of functions.
        *args : passed to constructors

    Returns:
        funcs : list of functions
        names : names of each function/model feature
    """
    
    funcs, names = [], []

    #### 
    poly_filter = fnmatch.filter(basis_library, "poly_*")
    if len(poly_filter) > 0:
        for _basis in poly_filter:
            order = int(_basis[5:])
            _f, _n = _construct_polynomial_basis_SINDYc(q, u, order, *args)
            funcs += _f
            names += _n

    sine_filter = fnmatch.filter(basis_library, "sinusoidal_*")
    if len(sine_filter) > 0:
        for _basis in sine_filter:
            order = int(_basis[11:])
            _f, _n = _construct_sinusoidal_basis_SINDYc(q, u, order, *args)
            funcs += _f
            names += _n

    if custom_library is not None:
        funcs += custom_library["functions"]
        names += custom_library["names"]

    if len(funcs) == 0:
        raise Exception("No valid library functions detected. Current options are [poly_*, sinusoidal_*], or specify a custom library dictionary.")
    
    return funcs, names

def _ODE_SINDYc(t : float,
                q : np.ndarray,
                t_rkh : states._array_like,
                drivers_itp : np.ndarray,
                Xi : np.ndarray,
                state : states.SindycState) -> np.ndarray:
    """
    Method for the SINDY model system of equations, passsed to Runge-Kutta pusher.

    Arguments:
        t : time point
        q : Array of shape [N,1] containing the current state of the solution q (initial condition)
        t_rkh : half-step time axis, used to get interpolated drivers
        drivers_itp : interpolated drivers, half-step resolution
        Xi : coefficient matrix for the time step
        state : ModelState variable

    Returns:
        dq_dt : Array of shape [N,1] containing the differential element of q(t)
    """
    theta = _SINDYc_basis_xform(q.squeeze(), drivers_itp[np.argmin(np.abs(t_rkh - t)),:], state)
    return Xi @ theta

def _SINDYc_basis_xform(q : states._array_like,
                        u : states._array_like, 
                        state : states.SindycState) -> np.ndarray:
    """
    Transforms the input data into the "theta" vector for the SINDYc model.

    Arguments:
        q : Contains solution-space data
        u : Contains drivers
        state : state variable

    Returns:
        the transformed data
    """

    return np.array([f(q,u) for f in state.feature_library]).reshape((len(state.feature_library), 1))

def _interpolate_drivers_SINDYc(t : states._array_like, 
                                tp : states._array_like, 
                                drivers: np.ndarray, 
                                state : states.SindycState) -> np.ndarray:
    """
    Pre-builds the interpolated sindy drivers.
    
    Arguments:
        t : Contains interpolated time axis
        tp : Contains original time axis
        drivers : array of size [len(tp), number of drivers], contains driver data
        state : state variable

    Returns:
        Interpolated drivers on the new time axis
    """

    return utils.interpolate_matrix(t, tp, drivers, state.interp_method, axis=0)

def _get_coeffs_SINDYc(kp : float, 
                       state : states.SindycState, 
                       kp_break : int = 3) -> np.ndarray:

    #### function for implementing the kp switch currently in the sindyc model
    #### TODO: ask about this. It seems to me the model is missing fundamental physics if this is necessary
    if kp < kp_break:
        return state.Xi[0] # "kp" model
    else:
        return state.Xi[1] # "f10" model

def _construct_polynomial_basis_SINDYc(q : states._array_like, 
                                       u : states._array_like,
                                       order : int,
                                       *args) -> list[states.FunctionType]:

    #### convenience
    Nq, Nu = len(q), len(u)

    #### linear
    if order == 1:
    
        #### construct all quadratic functions: q_i*q_j, u_i*u_j and q_i*u_j
        funcs = [lambda q, u, i=i: q[i] for i in range(Nq)] + \
                [lambda q, u, i=i: u[i] for i in range(Nu)]
        
        #### function labels
        names = [f"q{i}" for i in range(Nq)] + \
                [f"u{i}" for i in range(Nu)]
    
    #### quadratic
    elif order == 2:
        
        #### construct all quadratic functions: q_i*q_j, u_i*u_j and q_i*u_j
        funcs = [lambda q, u, i=i, j=j: q[i]*q[j] for i in range(Nq) for j in range(Nq)] + \
                [lambda q, u, i=i, j=j: u[i]*u[j] for i in range(Nu) for j in range(Nu)] + \
                [lambda q, u, i=i, j=j: q[i]*u[j] for i in range(Nq) for j in range(Nu)]
        
        #### function labels
        names = [f"q{i}*q{j}" for i in range(Nq) for j in range(Nq)] + \
                [f"u{i}*u{j}" for i in range(Nu) for j in range(Nu)] + \
                [f"q{i}*u{j}" for i in range(Nq) for j in range(Nu)]
        
    #### cubic
    elif order == 3:

        #### construct all quadratic functions: q_i*q_j, u_i*u_j and q_i*u_j
        funcs = [lambda q, u, i=i, j=j, k=k: q[i]*q[j]*q[k] for i in range(Nq) for j in range(Nq) for k in range(Nq)] + \
                [lambda q, u, i=i, j=j, k=k: u[i]*u[j]*u[k] for i in range(Nu) for j in range(Nu) for k in range(Nu)] + \
                [lambda q, u, i=i, j=j, k=k: q[i]*q[j]*u[k] for i in range(Nq) for j in range(Nq) for k in range(Nu)] + \
                [lambda q, u, i=i, j=j, k=k: q[i]*u[j]*u[k] for i in range(Nq) for j in range(Nu) for k in range(Nu)]
        
        #### function labels
        names = [f"q{i}*q{j}*q{k}" for i in range(Nq) for j in range(Nq) for k in range(Nq)] + \
                [f"u{i}*u{j}*u{k}" for i in range(Nu) for j in range(Nu) for k in range(Nu)] + \
                [f"q{i}*q{j}*u{k}" for i in range(Nq) for j in range(Nq) for k in range(Nu)] + \
                [f"q{i}*u{j}*u{k}" for i in range(Nq) for j in range(Nu) for k in range(Nu)]
    
    return funcs, names


def _construct_sinusoidal_basis_SINDYc(q : states._array_like, 
                                       u : states._array_like,
                                       order : int,
                                       *args) -> list[states.FunctionType]:
    
    if len(args) > 0:
        k = 2 * np.pi / args[0]
    else:
        k = 2 * np.pi
    
    #### convenience
    Nq, Nu = len(q), len(u)
    
    #### construct all quadratic functions: q_i*q_j, u_i*u_j and q_i*u_j
    funcs = [lambda q, u, i=i: np.sin(k * q[i])**order for i in range(Nq)] + \
            [lambda q, u, i=i: np.sin(k * u[i])**order for i in range(Nu)]
    
    #### function labels
    names = [f"sin^{order}(q{i})" for i in range(Nq)] + \
            [f"sin^{order}(u{i})" for i in range(Nu)]
    
    return funcs, names