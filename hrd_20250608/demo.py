# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: tiegcm_rope_env
#     language: python
#     name: python3
# ---

# +
#Latest version 2025-06-08

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

# +
forward_propagation = 1
init_date = pd.to_datetime('2023-05-05 18:00:00')

latitude_values = [-90., 89] # degrees
local_time_values = [20.7, 8] # hours
altitude_values = [570., 456] # km
lla_array = np.vstack((latitude_values, local_time_values, altitude_values)).T.reshape((-1, 3))
timestamps = [pd.to_datetime('2023-05-03 00:00:00'), pd.to_datetime('2023-05-03 00:00:00')]


sindy = rope.rope_propagator()
sindy.propagate_models(init_date = init_date, forward_propagation = forward_propagation)


rope_density = rope.rope_data_interpolator( data = sindy )

all_models, dmd_density, ensemble_density, density_std = rope_density.interpolate(timestamps, lla_array)

ensemble_density

# +
#To import CHAMP, GRACE-FO and SWARM data, use Filezilla at 
# thermosphere.tudelft.nl
# user: anonymous
# password:

import pandas as pd
import glob

columns = [
    'date', 'time', 'GPS', 'alt', 'lon', 'lat', 'lst', 'arglat',
    'accelerometer_density', 'dens_mean', 'flag_dens', 'flag_dens_mean'
]

file_paths = sorted(glob.glob('./champ_data/*.txt'))

df_list = []

for file_path in file_paths:
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        comment='#',
        names=columns
    )
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.set_index('datetime', inplace=True)
    df.drop(columns=['date', 'time', 'GPS'], inplace=True)
    
    df_list.append(df)

champ_all = pd.concat(df_list)

champ_all.sort_index(inplace=True)

columns = [
    'date', 'time', 'GPS', 'alt', 'lon', 'lat', 'lst', 'arglat',
    'accelerometer_density', 'dens_mean', 'flag_dens', 'flag_dens_mean'
]

file_paths = sorted(glob.glob('./grace_data/*.txt'))

df_list = []

for file_path in file_paths:
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        comment='#',
        names=columns
    )
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.set_index('datetime', inplace=True)
    df.drop(columns=['date', 'time', 'GPS'], inplace=True)
    
    df_list.append(df)

grace_all = pd.concat(df_list)

grace_all.sort_index(inplace=True)

columns = [
    'date', 'time', 'GPS', 'alt', 'lon', 'lat', 'lst', 'arglat',
    'accelerometer_density'
]

file_paths = sorted(glob.glob('./swarm_data/SA_*.txt'))

df_list = []

for file_path in file_paths:
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        comment='#',
        names=columns
    )
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.set_index('datetime', inplace=True)
    df.drop(columns=['date', 'time', 'GPS'], inplace=True)
    
    df_list.append(df)

swarma_all = pd.concat(df_list)

swarma_all.sort_index(inplace=True)

file_paths = sorted(glob.glob('./swarm_data/SC_*.txt'))

df_list = []

for file_path in file_paths:
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        comment='#',
        names=columns
    )
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.set_index('datetime', inplace=True)
    df.drop(columns=['date', 'time', 'GPS'], inplace=True)
    
    df_list.append(df)

swarmc_all = pd.concat(df_list)

swarmc_all.sort_index(inplace=True)

# +
msis_version = 2.1
omega = 45

def plot_densities(outputs_df, satellite_name = 'GRACE-FO', plot_name = '2023a'):
    fig, ax = plt.subplots(1, 1, figsize=(25, 10), sharex=False)

    # Subplot 1 — Density comparison
    ax.plot(outputs_df.datetime, outputs_df.accelerometer_density, label=f"{satellite_name} acceleromenter density", color='tab:blue', linewidth=2)
    ax.plot(outputs_df.datetime, outputs_df.msis, label="NRL-MSIS 2.1", color='pink', linewidth=1.0)
    ax.plot(outputs_df.datetime, outputs_df.ensemble_density, label="Ensemble density", color='orange', linewidth=1.5)
    ax.plot(outputs_df.datetime, outputs_df.debiased_ensemble_density, label="Debiased ensemble density", color='red', linewidth=1.)
    # ax.fill_between(timestamps2023b, debiased_ensemble_density2023b - density_std2023b, debiased_ensemble_density2023b + density_std2023b,
    #                    color='grey', alpha=0.2, label="Ensemble confidence interval")
    # ax.plot(timestamps2023b, dmd_density2023b, label="DMD", color='violet', linewidth=1.0)


    # Labels and legend
    ax.set_ylabel(r"$\rho$ (kg/m$^3$)", fontsize=14)
    ax.set_title(f"Neutral density comparison – {satellite_name} vs ROPE vs NRL-MSIS 2.1", fontsize=16)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    ax12 = ax.twinx()
    ax12.plot(outputs_df.datetime, outputs_df.kp_prop, label="$K_p$", color='tab:green', linestyle='--', linewidth=1)

    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=90))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_rotation(25)
        label.set_fontweight('bold')
        label.set_fontsize(12)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=25)
    # ax.set_ylim(0.15e-12, 5.2e-11)
    plt.tight_layout()
    plt.savefig(f'./imgs/{satellite_name.lower()}_vs_rope_ensemble_{plot_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

def bias_calculator(start_date, bias_propagation, dataset):

    start_debias = pd.to_datetime(start_date) - timedelta(hours= bias_propagation * 24 + delta_rho_ic + 1)
    end_debias = pd.to_datetime(start_date) - timedelta(hours= delta_rho_ic + 1)

    sindy = rope.rope_propagator(selected_bf_dict = selected_bf_dict, delta_rho_ic = delta_rho_ic)
    sindy.propagate_models(init_date = start_debias, forward_propagation = bias_propagation)
    rope_density_interpolator = rope.rope_data_interpolator( data = sindy )


    debias_dataset = dataset.loc[start_debias:end_debias]
    debias_dataset['hour'] = debias_dataset.index.hour
    debias_dataset['hour_minute'] = debias_dataset.index.minute/60 + debias_dataset['hour']
    debias_dataset['hms'] = debias_dataset.index.second + debias_dataset.index.minute*100 + debias_dataset['hour'] * 1000
    debias_dataset['t1'] = np.cos(np.pi*2.*debias_dataset['hms'].values/omega)
    debias_dataset['t2'] = np.sin(np.pi*2.*debias_dataset['hms'].values/omega)


    timestamps = debias_dataset.index.values
    latitude_values = debias_dataset.lat.values # degrees
    local_time_values = debias_dataset.lst.values # hours
    altitude_values = debias_dataset.alt.values/1000. # km
    lla_array = np.vstack((latitude_values, local_time_values, altitude_values)).T.reshape((-1, 3))


    _, _, ensemble_density, _ = rope_density_interpolator.interpolate(timestamps, lla_array)
    debias_dataset['debias_ratio'] = np.abs(debias_dataset.accelerometer_density/ensemble_density)
    return debias_dataset 

def build_output_data(start_date, end_date, interpolator, msis_version, dataset):
    
    subset = dataset.loc[start_date:end_date]
    
    timestamps = subset.index.values
    latitude_values = subset.lat.values # degrees
    local_time_values = subset.lst.values # hours
    longitude_values = subset.lon.values
    altitude_values = subset.alt.values/1000. # km
    lla_array = np.vstack((latitude_values, local_time_values, altitude_values)).T.reshape((-1, 3))

    all_models, dmd_density, ensemble_density, density_std = interpolator.interpolate(timestamps, lla_array)

    result = msis.calculate(
        timestamps,
        longitude_values,
        latitude_values,
        altitude_values, 
        geomagnetic_activity=-1,
        version = msis_version
    )

    msis_rho = result[:, 0]
    accelerometer_density = subset["accelerometer_density"].values

    t1 = interpolator.data.interval_hourly_drivers[1, :]
    t2 = interpolator.data.interval_hourly_drivers[2, :]
    t3 = interpolator.data.interval_hourly_drivers[3, :]
    t4 = interpolator.data.interval_hourly_drivers[4, :]
    kp = interpolator.data.interval_hourly_drivers[6, :]
    f10 = interpolator.data.interval_hourly_drivers[5, :]
    hourly_time_series = interpolator.data.hourly_date_series

    print(hourly_time_series.shape, kp.shape, f10.shape)
    print(hourly_time_series.min(), hourly_time_series.max())

    hourly_drivers_df = pd.DataFrame({
    'datetime': pd.to_datetime(hourly_time_series),
    'f10': f10,
    'kp': kp, 't1': t1, 't2': t2, 't3': t3, 't4': t4})
    hourly_drivers_df['datetime'] = pd.to_datetime(hourly_drivers_df['datetime'])

    outputs_df = pd.merge(
        pd.DataFrame(timestamps, columns=['datetime']),
        hourly_drivers_df,
        on='datetime',
        how='left'
    )
    print(interpolator.data.date_series.shape, interpolator.data.interval_interpolated_drivers[5, :].shape)
    interpolated_outputs = pd.DataFrame({'datetime': interpolator.data.date_series, 
        'f10_prop': interpolator.data.interval_interpolated_drivers[5, :], 
        'kp_prop': interpolator.data.interval_interpolated_drivers[6, :]})
    
    outputs_df = pd.merge(
        outputs_df,
        interpolated_outputs,
        on='datetime',
        how='left'
    )

    print(interpolator.data.interval_interpolated_drivers.shape, interpolator.data.date_series.shape, outputs_df.shape)
    outputs_df.sort_values('datetime', inplace=True)
    outputs_df.ffill(inplace=True)
    outputs_df.reset_index(drop=True, inplace=True)
    outputs_df['lst'] = local_time_values
    outputs_df['lat'] = latitude_values
    outputs_df['ensemble_density'] = ensemble_density
    outputs_df['accelerometer_density'] = accelerometer_density
    outputs_df['msis'] = msis_rho
    

    return timestamps, accelerometer_density, ensemble_density, dmd_density, \
        density_std, msis_rho, outputs_df, latitude_values, local_time_values


def run_demo(start_date, end_date, rope_density, msis_version, dataset, bias_propagation):

    timestamps, accelerometer_density, ensemble_density, dmd_density, \
        density_std, msis_rho, outputs_df, latitudes, lst_values = \
            build_output_data(start_date, end_date, rope_density, msis_version, dataset)
    
    debias_dataset = bias_calculator(start_date, bias_propagation, dataset)
    
    debias_ratio_df = debias_dataset.groupby(['t1', 't2']).agg(mean_ratio = ('debias_ratio', 'mean')).reset_index().copy()
    debias_density_df = pd.DataFrame({'datetime': timestamps, 
        'hour_minute': pd.to_datetime(timestamps).hour + pd.to_datetime(timestamps).minute/60., 
            'hms': pd.to_datetime(timestamps).second + pd.to_datetime(timestamps).minute*100 + pd.to_datetime(timestamps).hour*1000,
                'den': ensemble_density})
    debias_density_df['t1'] = np.cos(np.pi*2.*debias_density_df['hms'].values/omega)
    debias_density_df['t2'] = np.sin(np.pi*2.*debias_density_df['hms'].values/omega)

    debias_density_df = pd.merge(debias_density_df, debias_ratio_df, on=['t1', 't2'], how = 'left')
    debias_density_df['debiased_density'] = debias_density_df['den'] * debias_density_df['mean_ratio']
    outputs_df['debiased_ensemble_density'] = debias_density_df.debiased_density.values   

    return outputs_df

# Latest Ridge parameters are [1., 1000, 10000, 100000]
for alpha_ridge in [1]:
    selected_bf_dict = {
            'poly': 1.,
            # 'poly17': 5.,
            'poly12': 1.0,
            # 'poly1357': 100000, 
            # 'poly_sincos4': 1000, 
            # 'poly_sincos7': 1.,
            # 'poly_exp1': 1.0,
            # 'poly_exp2': 1,
            'poly_exp12': 1000,
            # 'poly_exp22': alpha_ridge
        }
    # for basis in all_bf_dict.keys():
    #     for alpha_ridge in [1, 5, 10, 100, 1000, 10000, 100000]:

    # selected_bf_dict = {
    #         basis: alpha_ridge
    #     }

    lst_bias = 0. #-2.0 is good
    alt_bias = 0.#40.
    delta_rho_ic = 0

    lt_low = 0
    lt_high = 23.66666667

    lat_low = -87.5
    lat_high = 87.5

    alt_low = 100
    alt_high = 980

    #Check if start and end dates are compatible with forward_propagation

    bias_propagation = 20


    start_date = "2023-01-01 12:00:00"
    end_date = "2023-01-05 18:00:00"
    forward_propagation = 5
    satellite_name = 'GRACE-FO'
    plot_name = '2023a'
    dataset = grace_all

    start_date = "2003-10-28 00:00:00"
    end_date = "2003-11-01 12:00:00"
    forward_propagation = 5
    satellite_name = 'CHAMP'
    plot_name = '2003'
    dataset = champ_all

    start_date = "2023-05-05 00:00:00"
    end_date = "2023-05-08 00:00:00"
    forward_propagation = 3
    satellite_name = 'GRACE-FO'
    plot_name = '2023b'
    dataset = grace_all

    start_date = "2024-05-10 00:00:00"
    end_date = "2024-05-13 00:00:00"
    forward_propagation = 3
    satellite_name = 'GRACE-FO'
    plot_name = '2024'
    dataset = grace_all

    sindy = rope.rope_propagator(selected_bf_dict = selected_bf_dict, delta_rho_ic = delta_rho_ic)
    sindy.propagate_models(init_date = start_date, forward_propagation = forward_propagation)
    rope_density = rope.rope_data_interpolator( data = sindy)

    outputs_df = run_demo(start_date, end_date, rope_density, msis_version, dataset, bias_propagation)
    plot_densities(outputs_df, satellite_name = satellite_name, plot_name = plot_name)
# -















