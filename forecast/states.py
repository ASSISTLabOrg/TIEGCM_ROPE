"""
Contains data structures.

Contact: Violet Player
Email: violet.player@noaa.gov
"""

#===================================== Imports =====================================#

from numpy import ndarray
from dataclasses import dataclass
from typing import Union
from types import FunctionType
from scipy.spatial._kdtree import KDTree as _kdt_type
from sklearn.preprocessing._data import StandardScaler as _scaler_type
from sklearn.decomposition._pca import PCA as _pca_type

_array_like = Union[ndarray, list, tuple]

#===================================== Model state variables =====================================#

@dataclass
class ModelState:
    """
    Generic model state data storage. Currently doesn't hold much, but I expect that to change as the number of models increases
    
    Attributes:
        model_name: Label for the model.
    """
    model_name : str

@dataclass
class SindycState(ModelState):
    """
    State data storage for the SINDYc model.

    Attributes:
        model_name [ModelState]: Label for the model.
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

# @dataclass
# class DmdcState(ModelState):
#     """
#     State data storage for the DMDc model.

#     Attributes:
#         model_name [ModelState]: Label for the model.
#     """

#===================================== Dimensionality state variables =====================================#
@dataclass
class GridState:
    """
    State data storage for the physics model grid.

    Attributes:
        x : 

    """
    x : _array_like
    y : _array_like
    z : _array_like
    tree : _kdt_type

@dataclass
class SpaceState:
    """
    State data storage for whatever vector space is relevant to the problem

    Attributes:
        space_name : name of the vector space
        grid : GridState variable

    """
    space_name : str
    grid : GridState

@dataclass
class PcaState(SpaceState):
    """
    State data storage for the PCA dimensionality reduction method.

    Attributes:
        space_name [SpaceState]: name of the vector space
        grid [SpaceState]: GridState variable
        scaler: StandardScaler object from sklearn
        PCA: PCA object from sklearn
    """
    
    scaler : _scaler_type
    PCA : _pca_type

# @dataclass 
# class CoaeState(SpaceState):
#     """
#     State data storage for the COAE (COnvolutional AutoEncoder) dimensionality reduction method.

#     Attributes:
#         space_name [SpaceState]: name of the vector space
#         grid [SpaceState]: GridState variable
#     """

#===================================== Job state variables =====================================#

@dataclass
class TaskState:
    task_id : int # task identifier
    parameters : dict # task-specific parameters; depends on model
    ModelState : ModelState # global solver state variable
    