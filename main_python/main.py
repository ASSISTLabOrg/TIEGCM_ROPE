# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: tiegcm_rope_env
#     language: python
#     name: python3
# ---

# %%
#Latest version 2025-05-01 aa
import numpy as np
from datetime import datetime, timedelta
import orekit
import time
import pymsis
from pymsis import msis
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
import utilities_ds as u
from orekit.pyhelpers import download_orekit_data_curdir, setup_orekit_curdir
from os import path
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.linalg import expm, logm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import rope_class_hrd as rope

# download_orekit_data_curdir( './orekit-data.zip' )  # Comment this out once this file has already been downloaded for repeated runs
vm = orekit.initVM()
setup_orekit_curdir( './' )

print ( 'Java version:', vm.java_version )
print ( 'Orekit version:', orekit.VERSION )

# %%
forward_propagation = 3
init_date = pd.to_datetime('2003-10-28 00:00:00')


sindy = rope.rope_propagator(drivers = None)
sindy.propagate_models(init_date = init_date, forward_propagation = forward_propagation)


n_examples = 10
initial_altitude = 100 # km

##################### Set interpolator and interpolation points #####################

latitude_values = [34. for _ in range(n_examples)] # degrees
local_time_values = [22.7 for _ in range(n_examples)] # hours
altitude_values = np.linspace(initial_altitude, 1000, n_examples)# [210.] # km

# latitude_values = [34.] # degrees
# local_time_values = [22.7] # hours
# altitude_values = [210.] # km
timestamps = init_date

rope_density = rope.rope_data_interpolator( data = sindy )
lla_array = np.vstack((latitude_values, local_time_values, altitude_values)).T.reshape((-1, 3))
all_models, dmd_density, ensemble_density, density_std = rope_density.interpolate(timestamps, lla_array)
print('Ensemble density')
print(ensemble_density) 
print('Ensemble density standard deviation')
print(density_std)
