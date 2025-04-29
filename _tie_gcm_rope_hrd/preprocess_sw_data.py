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
from pymsis import msis

# %%
file_path = './sw_inputs/SW_1957.csv'
df = pd.read_csv(file_path)
df

# %%
file_path = './sw_inputs/SW_1957.csv'
f10_name = 'F10.7_OBS'
df = pd.read_csv(file_path)

# Select relevant columns (KP1-KP8, AP1-AP8, and F10.7_ADJ)
kp_columns = [f'KP{i}' for i in range(1, 9)]
ap_columns = [f'AP{i}' for i in range(1, 9)]
f10_column = [f10_name]
selected_columns = ['DATE'] + kp_columns + ap_columns + f10_column

df = df[selected_columns]

# Convert DATE column to datetime format
df['DATE'] = pd.to_datetime(df['DATE'])

# Pivot the KP and AP values to long format
df_kp = df.melt(id_vars=['DATE'], value_vars=kp_columns, var_name='KP_Hour', value_name='kp')
df_ap = df.melt(id_vars=['DATE'], value_vars=ap_columns, var_name='AP_Hour', value_name='ap')

# Extract hour information (KP and AP values are 3-hourly)
df_kp['Hour'] = (df_kp['KP_Hour'].str.extract('(\d+)').astype(int) - 1) * 3
df_ap['Hour'] = (df_ap['AP_Hour'].str.extract('(\d+)').astype(int) - 1) * 3

# Merge KP and AP data
df_long = pd.merge(df_kp, df_ap, on=['DATE', 'Hour'], how='outer')

# Expand F10.7_ADJ to match hourly resolution (daily values)
df_f10 = df[['DATE', f10_name]].copy()
df_f10.rename(columns={f10_name:'f10'}, inplace = True)
df_f10['Hour'] = 0  # Daily values at midnight

# Merge with long dataframe
df_long = pd.merge(df_long, df_f10, on=['DATE', 'Hour'], how='outer')

# Create full datetime index
df_long['datetime'] = df_long['DATE'] + pd.to_timedelta(df_long['Hour'], unit='h')
df_long = df_long.drop(columns=['DATE', 'Hour'])

# Set datetime index and interpolate missing values
df_long = df_long.set_index('datetime').sort_index()

df_long

# %%



df_resampled = df_long.resample('h').interpolate()[['f10', 'kp']]

df_resampled['ut_c'] = np.cos(2*np.pi*np.copy(df_resampled.index.hour)/24.)
df_resampled['ut_s'] = np.sin(2*np.pi*np.copy(df_resampled.index.hour)/24.)
df_resampled['doy_c'] = np.cos(2*np.pi*np.copy(df_resampled.index.day_of_year)/365.25)
df_resampled['doy_s'] = np.sin(2*np.pi*np.copy(df_resampled.index.day_of_year)/365.25)
df_resampled['year'] = df_resampled.index.year

df_resampled = df_resampled[['year', 'ut_c', 'ut_s', 'doy_c', 'doy_s', 'f10', 'kp']]
df_resampled['kp'] = df_resampled['kp']/10

data = df_resampled['f10'].values

# Define outlier threshold using IQR (Interquartile Range Method)
q1, q3 = np.percentile(data, [10, 85])
iqr = q3 - q1
lower_bound = q1 - 2 * iqr
upper_bound = q3 + 2. * iqr

# Identify outliers
outliers = (data < lower_bound) | (data > upper_bound)

# Replace outliers with NaN for interpolation
data_cleaned = data.astype(float)
data_cleaned[outliers] = np.nan

# Interpolate missing values (outliers) using linear interpolation
indices = np.arange(len(data_cleaned))
valid_indices = np.where(~np.isnan(data_cleaned))[0]  # Indices of valid values
interpolated_values = np.interp(indices, valid_indices, data_cleaned[valid_indices])


df_resampled['f10'] = interpolated_values

df_resampled

# %%
df_resampled.loc[(df_resampled.index.year == 2003)]['f10'].plot()

# %%
df_resampled.loc[(df_resampled.index.year == 2003)]['f10'].plot()

# %%
df_resampled.loc[(df_resampled.index.year == 2003)]['f10'].plot()

# %%
df_resampled.to_csv('./tie_gcm_rope_deliverable/sw_inputs/sw_all_years_preprocessed.csv', index=True)

# %%

# %%

# %%
