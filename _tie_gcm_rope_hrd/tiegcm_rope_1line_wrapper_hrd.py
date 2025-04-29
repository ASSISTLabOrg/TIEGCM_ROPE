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
from datetime import datetime, timedelta
import orekit
import time
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

# download_orekit_data_curdir( '../orekit-data.zip' )  # Comment this out once this file has already been downloaded for repeated runs
vm = orekit.initVM()
setup_orekit_curdir( './' )

print ( 'Java version:', vm.java_version )
print ( 'Orekit version:', orekit.VERSION )

# %%
forward_propagation = 1
init_date = pd.to_datetime('2003-10-28 00:00:00')

sw_all_years = pd.read_csv('./sw_inputs/sw_all_years_preprocessed.csv', sep = ',').drop(columns = ['datetime'])
sw_drivers_all_years = np.copy(sw_all_years.values.T)


sindy = rope.rope_propagator(drivers = None)
sindy.propagate_models(init_date = init_date, forward_propagation = forward_propagation)

rope_density = rope.rope_data_interpolator( data = sindy )

latitude_values = [67., 67.] # degrees
local_time_values = [22.7, 22.7] # hours
altitude_values = [124., 1000.] # km
timestamps = init_date


lla_array = np.vstack((latitude_values, local_time_values, altitude_values)).T.reshape((-1, 3))
all_models, dmd_density, ensemble_density, density_std = rope_density.interpolate(timestamps, lla_array)
ensemble_density, density_std

# %% [markdown]
# **Optimal Ridge parameters, Averages of Model, Basis Functions, and MAPE. Train dataset. All conditions.**
#
# 2001 - 2008 dataset - Low Altitudes
#
# | **Model**   | **Basis Function** | **Optimal Ridge** | **Average MAPE (%)** |
# |-------------|---------------------|-------------------|-----------------------|
# | DMDc        | -----               | 0.0               | 4.80                  |
# | SINDYc      | poly                | 10.0              | 4.48                  |
# | SINDYc      | poly13              | 0.5               | 4.02                  |
# | SINDYc      | poly135             | 10.0              | 4.07                  |
# | SINDYc      | poly1357            | 0.5               | 4.11                  |
# | SINDYc      | poly7               | 0.0               | 8.97                  |
#

# %%
sw_all_years = pd.read_csv('./sw_inputs/sw_all_years_preprocessed.csv', sep=',').drop(columns=['datetime'])
sw_drivers_all_years = np.copy(sw_all_years.values.T)

init_date = pd.to_datetime('2003-10-29 00:00:00')

execution_times = []
max_propagation_days = 15
propagation_resolution = 1

for n, forward_propagation in enumerate(list(range(1, max_propagation_days + 1))[::propagation_resolution]):
    
    start_time = time.time()
    sindy = rope.rope_propagator()
    sindy.propagate_models(init_date = init_date, forward_propagation = forward_propagation)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    if n >= 1:
        slope = elapsed_time - execution_times[n-1]['execution_time_sec']
    else:
        slope = 0
    execution_times.append({
        'forward_propagation': forward_propagation,
        'execution_time_sec': elapsed_time,
        'slope': slope
    })
    print(f"Completed forward_propagation={forward_propagation} in {elapsed_time:.3f} seconds")

execution_times_df = pd.DataFrame(execution_times)

def exponential(x, a, b):
    return a * np.exp(b * x)

def linear(x, a, b):
    return a*x + b

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

xdata = execution_times_df['forward_propagation'].values
x_fit = np.linspace(min(xdata), max(xdata), 500)
ydata = execution_times_df['execution_time_sec'].values


exponential_optimal_parameters, _ = curve_fit(exponential, xdata, ydata, p0=(1, 0.001))
linear_optimal_parameters, _ = curve_fit(linear, xdata, ydata, p0=(1.5, 0.1))

y_exponential_fit = exponential(x_fit, *exponential_optimal_parameters)
y_linear_fit = linear(x_fit, *linear_optimal_parameters)

# Plot actual vs exponential
plt.plot(xdata, ydata, 'o-', label='Actual Execution Time')
# plt.plot(x_fit, y_exponential_fit, 'r--', label = \
#     f'Exponential Fit: {exponential_optimal_parameters[0]:.2f} * exp({exponential_optimal_parameters[1]:.2f} * x)')
# plt.plot(x_fit, y_linear_fit, 'g--', label = \
#     f'Linear Fit: {linear_optimal_parameters[0]:.2f} * x + {linear_optimal_parameters[1]:.2f}')
plt.xlabel(r'Propagation Steps (days)')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs Forward Propagation')
plt.legend()
plt.grid(True)

for i in range(0, len(execution_times_df)):
    if i % 2 != 0:
        continue
    x = xdata[i]
    y = ydata[i]
    slope = execution_times_df['slope'][i]
    plt.annotate(f'{slope:.2f}s/day', (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8, color='blue')
plt.tight_layout()
plt.xticks(np.arange(1, max_propagation_days + 1, propagation_resolution))
plt.yticks(np.arange(0, max(execution_times_df['execution_time_sec']) + 1, .2))
# plt.savefig('./imgs/execution_time_vs_forward_propagation.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
t_interp = pd.to_datetime('2003-10-29 00:00:00')

simulation_points = 12000
lat_values = np.linspace(-90, 90, simulation_points)
lt_values = np.linspace(0, 24, simulation_points)
alt_values = np.linspace(100, 1000, simulation_points)
lla_array_full = np.vstack((lat_values, lt_values, alt_values)).T
max_interp_samples = simulation_points
execution_times_interp = []

base_timestamp = pd.to_datetime("2003-10-29 00:00:00")
time_deltas = pd.to_timedelta(np.arange(simulation_points), unit="s")
timestamps_full = base_timestamp + time_deltas

for n, interp_points in enumerate(list(range(1, max_interp_samples + 1))[::100000]):
    timestamps = timestamps_full[:interp_points]
    lla_array = lla_array_full[:interp_points]

    start_time = time.time()
    rope_density = rope.rope_data_interpolator( data = sindy )
    interpolated_models2, density_dmd2, density2, density_std2 = rope_density.interpolate(timestamps, lla_array)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    if n >= 2:
        slope = elapsed_time - execution_times_interp[n-2]['execution_time_sec']
    else:
        slope = 0
    execution_times_interp.append({
        'interp_points': interp_points,
        'execution_time_sec': elapsed_time,
        'slope': slope
    })
    print(f"Interpolated {interp_points} points in {elapsed_time:.6f} seconds")
    
execution_times_interp_df = pd.DataFrame(execution_times_interp)

def exponential(x, a, b):
    return a * np.exp(b * x)

def linear(x, a, b):
    return a*x + b

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

xdata = execution_times_interp_df['interp_points'].values
x_fit = np.linspace(min(xdata), max(xdata), 500)
ydata = execution_times_interp_df['execution_time_sec'].values


exponential_optimal_parameters, _ = curve_fit(exponential, xdata, ydata, p0=(1, 0.001))
linear_optimal_parameters, _ = curve_fit(linear, xdata, ydata, p0=(0.1, 0.1))

y_exponential_fit = exponential(x_fit, *exponential_optimal_parameters)
y_linear_fit = linear(x_fit, *linear_optimal_parameters)

# Plot actual vs exponential
plt.plot(xdata, ydata, 'o-', label='Actual Execution Time')
plt.plot(x_fit, y_linear_fit, 'g--', label = \
    f'Linear Fit: {linear_optimal_parameters[0]:.3f} * x + {linear_optimal_parameters[1]:.2f}')
plt.xlabel(r'Interpolation points (absolute number)')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs Interpolation Points')
plt.legend()
plt.grid(True)

for i, interp_points in enumerate(list(execution_times_interp_df.interp_points)):
    if i % 25 != 0:
        continue
    x = xdata[i]
    y = ydata[i]
    slope = execution_times_interp_df['slope'][i]
    plt.annotate(f'{slope:.2f}s/10k-points', (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8, color='blue')
plt.tight_layout()
# plt.savefig('./imgs/execution_time_vs_interpolation_points.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
from datetime import datetime, timedelta

filtered_data = pd.read_pickle('density_measures_2003.pkl')
forward_propagation = 2


start_time = pd.to_datetime('2003-09-29 00:00:00')
plot_sindy(start_time, forward_propagation, filtered_data, None, step = 25000, altitude_limit = 1000)


# %%

# %%

# %%

# %%

# %%

# %%
def plot_sindy(start_time, forward_propagation, filtered_data, sw_drivers_all_years, step = 25000, altitude_limit = 450):
    day_of_year = start_time.day_of_year
    year = start_time.year
    end_time = start_time + timedelta(days=forward_propagation)
    selected_density_observations = filtered_data[(filtered_data['datetime'] >= start_time) & 
                                (filtered_data['datetime'] <= end_time)].reset_index()
    selected_density_observations['date'] = selected_density_observations.datetime.dt.date
    selected_density_observations['hour'] = selected_density_observations.datetime.dt.hour

    for group in selected_density_observations.groupby('satellite'):
        group[1].reset_index(drop = True).index
        selected_density_observations.loc[selected_density_observations.satellite == group[0], 'T'] = 10*np.arange(group[1].reset_index().index[0], group[1].reset_index().index[-1]+1)


    selected_density_observations = selected_density_observations.rename(columns={
        'Geodetic Altitude (km)': 'alt',
        'Geodetic Latitude (deg)': 'lat',
        'Geodetic Longitude (deg)': 'lon',
        'Local Solar Time (hours)': 'lt',
        "Density_New (kg/m^3)": "new_density",
        "Density_Eric (kg/m^3)": "eric_density"
    })

    beta_sigma = 1

    selected_density_observations = selected_density_observations.loc[(selected_density_observations.alt>=100) & \
        (selected_density_observations.alt<=altitude_limit) & \
            (selected_density_observations['T'] <= float(60*(60*24*forward_propagation-1)))]
    
    T_values = selected_density_observations["T"].values
    timestamps = selected_density_observations['datetime'].values
    lat_values = selected_density_observations["lat"].values
    lt_values = selected_density_observations["lt"].values
    alt_values = selected_density_observations["alt"].values

    sindy = rope.rope_propagator(drivers = sw_drivers_all_years)
    sindy.propagate_models(init_date = start_time, forward_propagation = forward_propagation)

    print(sindy.t.max())
    rope_density = rope.rope_data_interpolator( data = sindy )

    lla_array = np.vstack((lat_values, lt_values, alt_values)).T.reshape((-1, 3))
    _, _, density2, density_std2 = rope_density.interpolate(timestamps, lla_array)

    selected_density_observations.loc[:, "sindy_density"] = density2
    selected_density_observations.loc[:, "sindy_std"] = density_std2

    interpolated_sindy = selected_density_observations.copy()

    beta_sigma = 1

    interpolated_sindy['sindy_avg_unbiased'] = interpolated_sindy['sindy_density'] - np.median(interpolated_sindy['sindy_density'] - interpolated_sindy['new_density'])
    interpolated_sindy['accelerometer derived density estimate'] = interpolated_sindy['new_density']
    interpolated_sindy['sindy_ci_lower_bound'] = interpolated_sindy['sindy_density'] - beta_sigma * interpolated_sindy['sindy_std']
    interpolated_sindy['sindy_ci_upper_bound'] = interpolated_sindy['sindy_density'] + beta_sigma * interpolated_sindy['sindy_std']


    interpolated_sindy['mpe_new'] = (interpolated_sindy['sindy_density'] - interpolated_sindy['new_density']) / interpolated_sindy['new_density']
    interpolated_sindy['mape_new'] = np.abs(interpolated_sindy['sindy_density'] - interpolated_sindy['new_density']) / interpolated_sindy['new_density']
    interpolated_sindy['mpe_eric'] = (interpolated_sindy['sindy_avg_unbiased'] - interpolated_sindy['eric_density'] ) / interpolated_sindy['eric_density']

    interpolated_sindy['mape_new'].mean(), interpolated_sindy['mpe_new'].mean()

    def kp_to_ap(kp):
        """Convert Kp index to Ap index using interpolation."""
        kp_values = np.array([0.0, 0.3, 0.7, 1.0, 1.3, 1.7, 2.0, 2.3, 2.7, 3.0, 3.3, 3.7, 4.0, 4.3, 4.7, 
                            5.0, 5.3, 5.7, 6.0, 6.3, 6.7, 7.0, 7.3, 7.7, 8.0, 8.3, 8.7, 9.0])
        ap_values = np.array([0, 2, 3, 4, 5, 6, 7, 9, 12, 15, 18, 22, 27, 32, 39, 48, 56, 67, 80, 94, 111, 
                            132, 154, 179, 207, 236, 300, 400])
        return np.interp(kp, kp_values, ap_values)

    agg_results = interpolated_sindy.groupby(['date', 'hour']).agg(new_density = ('new_density', 'median'), \
        eric_density = ('eric_density', 'median'), sindy = ('sindy_density', 'mean'), 
        lon = ('lon', 'last'),  # Take last longitude
        lat=('lat', 'last'),  # Take last latitude
        alt=('alt', 'last')).copy().reset_index()


    date_series = pd.date_range(start=start_time, end = end_time, freq='1h')[:(-1)]
    date_series.shape

    t0 = day_of_year * 24 - 24
    forward_hours = forward_propagation * 24
    tf = t0 + forward_hours
    delta_rho_ic = 6
    IC_idx = np.where(sindy.drivers[0,:] == year)[0]
    biased_ic_indices = np.arange(np.min(IC_idx) - delta_rho_ic, np.min(IC_idx) + tf)

    f10_series_msis = (sindy.drivers[:, IC_idx][5, t0:tf])
    kp_series_msis = (sindy.drivers[:, IC_idx][6, t0:tf])
    ap_series_msis = kp_to_ap(kp_series_msis)
    lon_msis = (agg_results.lon)
    lat_msis = (agg_results.lat)
    alt_msis = (agg_results.alt)

    agg_results['f10'] = f10_series_msis
    agg_results['kp'] = kp_series_msis
    agg_results['norm_f10'] = agg_results['f10']/agg_results['f10'].sum()

    num_steps = agg_results.shape[0]
    ap_series = np.copy(ap_series_msis)

    aps = np.zeros((num_steps, 7))

    aps[:, 0] = ap_series
    aps[:, 1] = np.roll(ap_series, 1)
    aps[0, 1] = ap_series[0]
    aps[:, 2] = np.roll(ap_series, 2)
    aps[:2, 2] = ap_series[0]
    aps[:, 3] = np.roll(ap_series, 3)
    aps[:3, 3] = ap_series[0]

    cumsum = np.cumsum(ap_series)
    cumsum = np.insert(cumsum, 0, 0) 

    start_indices = np.maximum(0, np.arange(num_steps) - 16)
    end_indices = np.maximum(0, np.arange(num_steps) - 8)
    window_sizes = np.maximum(1, end_indices - start_indices)
    window_sums = cumsum[end_indices + 1] - cumsum[start_indices]
    aps[:, 4] = window_sums / window_sizes
    aps[start_indices == 0, 4] = ap_series[0]

    start_indices = np.maximum(0, np.arange(num_steps) - 24)
    end_indices = np.maximum(0, np.arange(num_steps) - 16)
    window_sizes = np.maximum(1, end_indices - start_indices)
    window_sums = cumsum[end_indices + 1] - cumsum[start_indices]
    aps[:, 5] = window_sums / window_sizes
    aps[start_indices == 0, 5] = ap_series[0]

    start_indices = np.maximum(0, np.arange(num_steps) - 240)
    window_sizes = np.arange(1, num_steps + 1)
    window_sums = cumsum[1:] - cumsum[start_indices]
    aps[:, 6] = window_sums / window_sizes
    aps[start_indices == 0, 6] = ap_series[0]

    result = msis.run(
        dates=date_series.values,
        lons=lon_msis.values,
        lats=lat_msis.values,
        alts=alt_msis.values,
        f107s=f10_series_msis,
        aps=aps
    )

    msis_rho = result[:, 0]
    agg_results['msis'] = msis_rho
    agg_results[['ap1', 'ap2', 'ap3', 'ap4', 'ap5', 'ap6', 'ap7']] = aps

    df_to_plot = agg_results.iloc[:, :].copy()
    df_to_plot.to_pickle('comparison.pkl')

    beta_sigma = 0.8

    merged_df = pd.merge(interpolated_sindy, agg_results.drop(columns=['alt', 'lat', 'lon']), on=['date', 'hour'], how='left')

    merged_df['sindy_ci_lower_bound'] = merged_df['sindy_density'] - beta_sigma * merged_df['sindy_std']
    merged_df['sindy_ci_upper_bound'] = merged_df['sindy_density'] + beta_sigma * merged_df['sindy_std']

    minutes_resolution = 6
    hour_resolution = 60 * minutes_resolution

    T_values = pd.to_datetime(merged_df["datetime"])
    f10_series_msis = merged_df.f10
    kp_series_msis = merged_df.kp
    ap_series_msis = kp_to_ap(kp_series_msis)
    lon_msis = (merged_df['lon'])
    lat_msis = (merged_df.lat)
    alt_msis = (merged_df.alt)

    aps = merged_df[['ap1', 'ap2', 'ap3', 'ap4', 'ap5', 'ap6', 'ap7']].values

    total_elements = len(T_values)
    results = []
    batch_size = 50000

    for start in range(0, total_elements, batch_size):
        end = min(start + batch_size, total_elements)

        result_batch = msis.run(
            dates=T_values.values[start:end],
            lons=lon_msis.values[start:end],
            lats=lat_msis.values[start:end],
            alts=alt_msis.values[start:end],
            f107s=f10_series_msis[start:end],
            aps=aps[start:end]
        )
        
        results.append(result_batch)

    result = np.concatenate(results, axis=0)

    msis_rho = result[:, 0]
    merged_df['msis'] = msis_rho

    satellites = merged_df.satellite.unique()

    for satellite in satellites:
        satellite_df = merged_df.loc[merged_df.satellite == satellite].copy()
        # print(f'Mean lat: {satellite_df.lat.mean():.2f}, mean lon: {satellite_df.lon.mean():.2f}, mean alt: {satellite_df.alt.mean():.2f}')
        total_rows = len(satellite_df)
        for start in range(0, total_rows, step):
            end = min(start + step, total_rows)

            fig, ax1 = plt.subplots(figsize=(10, 5))

            satellite_df.iloc[start:end].plot(x='datetime', 
                                        y=['accelerometer derived density estimate', 'sindy_density', 'msis'], 
                                        ax=ax1, label=['Accelerometer Density', 'SINDY Density', 'MSIS'])
            ax1.set_ylabel(r'$\rho \left(\frac{kg}{m^3} \right)$', rotation = 0, labelpad=20)
            ax1.set_xlabel(r'Datetime')
            ax1.grid()

            x_vals = mdates.date2num(satellite_df.iloc[start:end]['datetime'])

            ax1.fill_between(
                x_vals,
                satellite_df.iloc[start:end]['sindy_ci_lower_bound'],
                satellite_df.iloc[start:end]['sindy_ci_upper_bound'],
                color='orange', alpha=0.3, label="SINDY Confidence Interval"
            )

            ax1.xaxis_date()
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

            ax2 = ax1.twinx()
            satellite_df.iloc[start:end].plot(x='datetime', 
                                        y=['norm_f10', 'kp'], 
                                        ax=ax2, linestyle='--', color=['green', 'red'], 
                                        label=['Normalized F10', 'Kp'])
            ax2.set_ylabel('Kp and Normalized F10')

            plt.title(fr"Plot for {satellite}, mean altitude = {satellite_df.iloc[start:end]['alt'].mean():.2f}, mean $k_p$ = {satellite_df.iloc[start:end]['kp'].mean():.2f}")

            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax1.tick_params(axis='x', labelsize=12)
            for label in ax1.get_xticklabels():
                label.set_fontsize(12)
                label.set_fontweight('bold')
            plt.savefig(f'./imgs/msis_comp_{satellite}_{start}.png', dpi = 300)
            plt.show()

# %%
idx = 5
year = 2003

f10_original = np.copy(sindy.original_drivers[:, sindy.original_drivers[0, :] == year][idx, :])
f10_celestrack = np.copy(sw_drivers_all_years[:, sw_drivers_all_years[0, :] == year][idx, :])

x = np.arange(len(f10_original))

diff = f10_original - f10_celestrack
difference_indices = np.where(diff != 0)[0]

plt.figure(figsize=(15, 6))
plt.plot(x, diff, label='difference', alpha=0.7)
plt.plot(x, f10_original, label='f10_original', alpha=0.7)
plt.plot(x, f10_celestrack, label='f10_celestrack', alpha=0.7)

plt.title('Comparison of f10_original vs f10_celestrack')
plt.xlabel('Index')
plt.ylabel('F10 Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%

# %%
