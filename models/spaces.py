"""
Contains the methods for manipulating various vector spaces

Public Methods:

    interpolate: 

Contact: Violet Player
Email: violet.player@noaa.gov
"""

#===================================== Imports =====================================#

import numpy as np
import states

#===================================== Public methods =====================================#

def interpolate(xn : states._array_like,
                q : np.ndarray,
                DimensionState : states.SpaceState,
                method : str = "knn",
                k : int = 8,
                *args) -> float:
    """
    Outer method for interpolating results in reduced space.

    Arguments:
        xn : Array-like of length(dims) point in configuration space to interpolate to
        q : ndarray of length(components) PCA component amplitude, already at correct time
        DimensionState : StateVar DimensionState, contains all the necessary matrices and vectors to run this
        k : number of points to retrieve from KDTree. Defaults to 8 for 3D cube.
        args : additional arguments for different interpolation methods.

    Returns:
        Interpolated datum.
    """
    
    #### PCA reduced-space interpolation
    if type(DimensionState) == states.PcaState:

        return _interpolate_PCA(xn, q, DimensionState, method, k, *args)
    
    else:

        raise Exception("DimensionState not reconized. Must be one of [PcaState]")

#===================================== PCA methods =====================================#

def _interpolate_PCA(xn : states._array_like, 
                     q : np.ndarray,
                     PCA : states.PcaState,
                     method : str = "knn",
                     k : int = 8,
                     *args) -> float:
    """
    Method for interpolated PCA-space data without needing to do the full xform.

    Arguments:
        xn : Array-like of length(dims) point in configuration space to interpolate to
        q : ndarray of length(components) PCA component amplitude, already at correct time
        PCA : StateVar PcaState, contains all the necessary matrices and vectors to run this
        k : number of points to retrieve
        args : additional arguments for different interpolation methods.

    Returns:
        Interpolated datum.
    
    TODO: currently only supports KNN. Would like to add trilinear, tricubic, whatever...
    """

    #### query the tree for best indices
    d, indices = PCA.tree.query(xn, k=k)
    
    #### if requested point is on grid, no interpolation is required
    if d[0] == 0:

        #### inverse PCA transformation in the reduced index space
        Xp = np.squeeze(np.matmul(q.reshape((1,-1)), PCA.pca.components_[:,indices[0]]))

        #### inverse scaling transformation in the reduced index space
        #### and this is the final result
        X_itp = PCA.scaler.mean_[indices[0]] + PCA.scaler.scale_[indices[0]] * Xp

    else:

        #### inverse PCA transformation in the reduced index space
        Xp = np.squeeze(np.matmul(q.reshape((1,-1)), PCA.pca.components_[:,indices]))

        #### inverse scaling transformation in the reduced index space
        X = PCA.scaler.mean_[indices] + PCA.scaler.scale_[indices] * Xp

        #### interpolate based on requested method
        if method.lower()=="knn":
            X_itp = np.sum(X / d) / np.sum(1 / d) # inverse distance weighting
        
        else:
            raise Exception("Only supports the following interpolation methods: [knn]")
        
    return X_itp
