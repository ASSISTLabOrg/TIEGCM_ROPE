# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import orekit
import utilities_ds as u
from orekit.pyhelpers import download_orekit_data_curdir, setup_orekit_curdir
from os import path
import numpy as np
import pandas as pd
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.bodies import BodyShape, GeodeticPoint, OneAxisEllipsoid
from org.orekit.frames import Frame, FramesFactory, KinematicTransform
from org.orekit.models.earth.atmosphere import PythonAtmosphere
from org.orekit.time import AbsoluteDate, TimeScalesFactory, UTCScale
from org.orekit.utils import Constants, IERSConventions, PVCoordinates
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import interp1d
from scipy.linalg import expm, logm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import rope_class as rope

# download_orekit_data_curdir( '../orekit-data.zip' )  # Comment this out once this file has already been downloaded for repeated runs
vm = orekit.initVM()
setup_orekit_curdir( './' )

print ( 'Java version:', vm.java_version )
print ( 'Orekit version:', orekit.VERSION )

# %%
######################################################## Inputs section ########################################################
init_date_str = '2003-11-03 00:00:00'
timestamps = np.array(['2003-11-03 04:00:00', '2003-11-03 10:00:00'])
lla_array = np.array([[87., 23.7, 440.5], [87., 23.7, 440.5]])
forward_propagation = 6

sw_all_years = pd.read_csv('./sw_inputs/sw_all_years_preprocessed.csv', sep = ',').drop(columns = ['datetime'])
sw_drivers_all_years = np.copy(sw_all_years.values.T)

sindy = rope.rope_data(drivers = sw_drivers_all_years)
sindy.propagate_models(init_date = init_date_str, forward_propagation = forward_propagation)
rope_density = rope.rope_data_interpolator( data = sindy )


density_poly, density_poly_all, density_dmd, density, density_std = \
    rope_density.interpolate(timestamps, lla_array)
print({'density': density, 'standard deviation': density_std})
