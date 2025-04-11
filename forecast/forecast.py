"""
Contains the methods needed to run a trained forecast.

Public Methods:

    forecast: 

Contact: violet.player@noaa.gov
"""

#### generic python
import numpy as np
import multiprocessing as mproc

#### module imports
import forecast.models as md
import forecast.statevars as sv

#===================================== Public methods =====================================#

def forecast(tasks: list[sv.TaskState], nproc=1) -> list:

    #### forecasting model selection
    #### TODO: currently, only supports one *type* of model. In principal this should be formatted to accept tasks with different states
    if type(tasks[0].ModelState) == sv.SindycState:
        # TODO: forecaster check
        #md._run_sindyc_check(state, tasks) # verify the inputs are correctly formatted
        fcst_model = md.SINDYc_forecast # assign forecast model

    elif type(tasks[0].ModelState) == sv.DmdcState:
        # models._run_dmdc_check(state)
        # fcst_model = models.linear_forecast_dmd
        raise Exception("Not built yet.")
    
    else:
        raise Exception("Forecast model not recognized.")
    

    #### Parallel & serial job processing
    if nproc > 1:
        p = mproc.Pool(nproc)
        output = p.map_async(fcst_model, tasks)
        p.close()
        p.join()
        #results = _format_results(output, parallel=True)

    else:
        output = []
        for task in tasks:
           output.append(fcst_model(task))
        #results = _format_results(output)

    #return results
    # TODO: output formatting?
    return output

#===================================== post-processing functions =====================================#

def _format_results(output: list, 
                    parallel=False):
    return None
