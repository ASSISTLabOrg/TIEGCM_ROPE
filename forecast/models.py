#### generic python
import numpy as np

#### module imports
import forecast.statevars as sv
import forecast.utils as utils

#===================================== SINDY model functions =====================================#

def SINDYc_forecast(job : sv.TaskState):
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
    fcst = np.zeros((len(t_rkf), state.q0.size))
    fcst[0,:] = state.q0
    
    #### Forecast!
    for i in range(1,len(t_rkf)):
        Xi = _get_coeffs_SINDYc(drivers[i,state.indices["kp"]], state)
        fcst[i,:] = _RKO4_step(_ODE_SINDYc, 
                              fcst[i-1,:].reshape((-1,1)),
                              t_rkf[i],
                              state.dt,
                              *[t_rkh, drivers_itp, Xi, state]).squeeze()
        
    if job.parameters["forecast_times"] is not None:
        fcst = utils.interpolate_matrix(job.parameters["forecast_times"], t_rkf, 
                                        fcst, state.interp_method, axis=0)

    return fcst

def _ODE_SINDYc(t : float,
                q : np.ndarray,
                t_rkh : sv._array_like,
                drivers_itp : np.ndarray,
                Xi : np.ndarray,
                state : sv.SindycState) -> np.ndarray:
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

def _SINDYc_basis_xform(q : sv._array_like,
                        u : sv._array_like, 
                        state : sv.SindycState) -> np.ndarray:
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

def _interpolate_drivers_SINDYc(t : sv._array_like, 
                                tp : sv._array_like, 
                                drivers: np.ndarray, 
                                state : sv.SindycState) -> np.ndarray:
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
                       state : sv.SindycState, 
                       kp_break : int = 3) -> np.ndarray:

    #### function for implementing the kp switch currently in the sindyc model
    #### TODO: ask about this. It seems to me the model is missing fundamental physics if this is necessary
    if kp < kp_break:
        return state.Xi[0] # "kp" model
    else:
        return state.Xi[1] # "f10" model
    
def construct_basis_SINDYc(q : sv._array_like, 
                           u : sv._array_like,
                           basis_library=["linear"],
                           custom_library=None) -> list[sv.FunctionType]:
    """
    
    """
    
    funcs, names = [], []
    if "linear" in basis_library:
        _f, _n = _construct_linear_basis_SINDYc(q, u)
        funcs += _f
        names += _n

    if "quadratic" in basis_library:
        _f, _n = _construct_quadratic_basis_SINDYc(q, u)
        funcs += _f
        names += _n

    if custom_library is not None:
        funcs += custom_library["functions"]
        names += custom_library["names"]

    if len(funcs) == 0:
        raise Exception("No valid library functions detected. Current options are [linear, quadratic]")
    
    return funcs, names
    
def _construct_linear_basis_SINDYc(q : sv._array_like, 
                                  u : sv._array_like) -> list[sv.FunctionType]:

    #### convenience
    Nq, Nu = len(q), len(u)
    
    #### construct all quadratic functions: q_i*q_j, u_i*u_j and q_i*u_j
    funcs = [lambda q, u, i=i: q[i] for i in range(Nq)] + \
            [lambda q, u, i=i: u[i] for i in range(Nu)]
    
    #### function labels
    names = [f"q{i}" for i in range(Nq)] + \
            [f"u{i}" for i in range(Nu)]
    
    return funcs, names

def _construct_quadratic_basis_SINDYc(q : sv._array_like, 
                                     u : sv._array_like) -> list[sv.FunctionType]:
    """
    Documentation...
    
    """
    #### convenience
    Nq, Nu = len(q), len(u)
    
    #### construct all quadratic functions: q_i*q_j, u_i*u_j and q_i*u_j
    funcs = [lambda q, u, i=i, j=j: q[i]*q[j] for i in range(Nq) for j in range(Nq)] + \
            [lambda q, u, i=i, j=j: u[i]*u[j] for i in range(Nu) for j in range(Nu)] + \
            [lambda q, u, i=i, j=j: q[i]*u[j] for i in range(Nq) for j in range(Nu)]
    
    #### function labels
    names = [f"q{i}*q{j}" for i in range(Nq) for j in range(Nq)] + \
            [f"u{i}*u{j}" for i in range(Nu) for j in range(Nu)] + \
            [f"q{i}*u{j}" for i in range(Nq) for j in range(Nu)]

    return funcs, names

#===================================== DMD model functions =====================================#

def forecast_dmdc(state):
    return None

def _ODE_dmdc(state):
    return None

#===================================== General model functions =====================================#

def _RKO4_step(func : sv.FunctionType, 
               y0 : sv.Union[float, np.ndarray],
               t0 : float, 
               h : float, 
               *args) -> sv.Union[float, np.ndarray]:
    """
    Takes a single step in a Runge-Kutta O(4) solver for equations of the form dy/dt = f(t,y).
    Makes four function calls per time step.

    Presumes you will do a good job keeping function output shapes consistent.
    TODO: some amount of case handling/reshaping? Is that needed?

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
