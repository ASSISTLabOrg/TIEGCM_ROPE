"""
Contains the methods needed to run a trained forecast.

Public Methods:

    forecast: 

Contact: Violet Player
Email: violet.player@noaa.gov
"""

#===================================== Imports =====================================#

import numpy as np
import multiprocessing as mproc

import models.models
import models.states

#===================================== Public methods =====================================#

def forecast():

    return None


# def forecast():

#     mproc.set_forkserver_preload

#     return None

# def forecast(tasks: list[states.TaskState], 
#              pool=None,
#              nproc=1, 
#              close=False,
#              join=True,
#              verbose=False) -> list:
#     """
    
#     Arguments:
#         tasks : list of TaskState objects. Each task is passed to a seperate process in the async map
#         pool : multiprocessing pool. If none, nproc is used to build a new pool if > 1.
#         nproc : number of processes. If pool is provided, this argument is ignored. If == 1, serial mode.
#         close : close the pool out.
#         join : joins the pool - function will hang on the main process until all workers are done computing.
#         verbose : print output?

#     Returns:
#         output : results of the calculations in the various TaskStates
        

#     """

#     #### error handling
#     if close and not join:
#         raise Exception("Cannot close pool without joining, or it won't close processess properly.")

#     #### TODO: currently, only supports one *type* of model. In principal this should be formatted to accept tasks with different states
#     if type(tasks[0].ModelState) == states.SindycState:
#         # TODO: forecaster check
#         #models._run_sindyc_check(state, tasks) # verify the inputs are correctly formatted
#         fcst_model = models.SINDYc_forecast # assign forecast model

#     elif type(tasks[0].ModelState) == states.DmdcState:
#         # models._run_dmdc_check(state)
#         # fcst_model = models.linear_forecast_dmd
#         raise Exception("Not built yet.")
    
#     else:
#         raise Exception("Forecast model not recognized.")
    

#     #### Parallel processing
#     if (nproc > 1) or (pool is not None):

#         #### open a new pool if not provided
#         if pool is None:
#             if verbose: print(f"Opening a pool with {nproc} processes...")
#             pool = mproc.Pool(nproc)

#         if verbose: print("Running forecasts...")
#         output = pool.map_async(fcst_model, tasks)

#         #### will hang here until all computations are completed.
#         if join:
#             pool.join()
#             if verbose: print("All tasks completed.")

#         #### close out pool
#         if close:
#             pool.close()
#             if verbose: print("Pool closed.")
        
#         #TODO: formatting
#         #results = _format_results(output, parallel=True)ublic methods

#     else:
#         output = []
#         for task in tasks:
#            output.append(fcst_model(task))
        
#         if verbose: print("All tasks completed.")
        
#         #TODO: formatting
#         #results = _format_results(output)

#     #return results
#     return output

#===================================== post-processing functions =====================================#

def _format_results(output: list, 
                    parallel=False):
    return None
