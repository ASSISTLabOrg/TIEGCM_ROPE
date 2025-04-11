import numpy as np
from dataclasses import dataclass
from typing import Union
from types import FunctionType

_array_like = Union[np.ndarray, list, tuple]

#===================================== Forecaster state variables =====================================#

@dataclass
class ModelState:
    """
    Generic model state data storage.

    TODO: Currently doesn't hold much, but I expect that to change as the number of models increases
    """

    #### name of the model instance
    model_name : str

@dataclass
class SindycState(ModelState):
    """
    State data storage for the SINDYc model

    Attributes:

        Xi : coefficient matrices. Currently expects list/tuple containining (Xi_1, Xi_2)
        feature_library : library of model functions
        feature_names : names of functions; purely for inspection
        q0 : initial state of solution
        indices : relevant indices
        interp_method : how to interpolate matrices
        dt : time step of runge-kutta method
    """

    Xi : _array_like
    feature_library : list[FunctionType]
    feature_names : list[str]
    q0 : _array_like
    indices : dict
    interp_method : str
    dt : float

@dataclass
class DmdcState(ModelState):
    """
    State data storage for the DMDc model

    Attributes:

        Xi : coefficient matrices. Currently expects list/tuple containining (Xi_1, Xi_2)
        feature_library : library of model functions
        feature_names : names of functions; purely for inspection
        q0 : initial state of solution
        indices : relevant indices
        interp_method : how to interpolate matrices
        dt : time step of runge-kutta method
    """
    attribute : None
    # Xi : _array_like
    # feature_library : list[FunctionType]
    # feature_names : list[str]
    # q0 : _array_like
    # indices : dict
    # interp_method : str
    # dt : float


#===================================== Dimensionality state variables =====================================#

@dataclass
class DimensionState:
    attribute : None

@dataclass
class PcaState(DimensionState):
    Usvd : np.ndarray

@dataclass 
class CoaeState(DimensionState):
    attribute : None

#===================================== Job state variables =====================================#

@dataclass
class TaskState:
    task_id : int # task identifier
    parameters : dict # task-specific parameters; depends on model
    ModelState : ModelState # global solver state variable
    