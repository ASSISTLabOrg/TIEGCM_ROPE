import h5py
from scipy.io import loadmat
import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat
from scipy.linalg import logm
import glob
import re
import datetime as dt
from itertools import product, combinations_with_replacement
import warnings
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import interp1d


def matlab_datenum_to_py_datetime(dn):
    dn = np.asarray(dn, dtype=float)
    out = []
    for x in dn.flatten():
        day = int(x)
        frac = x - day
        date = dt.date.fromordinal(day - 366)
        seconds = frac * 86400.0
        out.append(dt.datetime.combine(date, dt.time()) + dt.timedelta(seconds=seconds))
    return np.array(out, dtype=object).reshape(dn.shape)

def add_lags(data, lags=[1, 2, 3]):
    "Constructs the lagged version of the input vector, with lags the input lags list. If, for instance,"
    "there is a pair of rows with shapes [2, m_times], then the output will have shape [2 + 2*len(lags), m_times]"
    if lags[0] == 0:
        return data.T 
    
    n_samples, n_features = data.T.shape
    lagged_data = []

    for lag in lags:
        padded = np.zeros((n_samples, n_features))
        padded[lag:] = data.T[:-lag]
        lagged_data.append(padded)

    return np.hstack([data.T] + lagged_data).T

def load_mat_any(path):
    # MATLAB v7.3 files are HDF5 and start with the HDF5 signature: b"\x89HDF\r\n\x1a\n"
    with open(path, "rb") as f:
        sig = f.read(8)

    is_hdf5 = (sig == b"\x89HDF\r\n\x1a\n")

    if is_hdf5:
        # v7.3: use h5py
        f = h5py.File(path, "r")
        return f, "h5py"  # remember to close f when done
    else:
        # pre-v7.3: use scipy
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        return data, "loadmat"
    
def build_ar_inputs(u_train, z1_train, z2_train, u_test, ar_lags, n_components, f10_idx, kp_idx, ar_drivers_flag, ar_state_flag):
    if ar_drivers_flag is False:
        u_train_ar = np.copy(u_train)
        u_test_ar = u_test

    else:
        u_train_ar = np.concatenate([np.copy(u_train), \
            add_lags(np.copy(u_train)[[f10_idx, kp_idx], :], lags = ar_lags)[2:, :]], axis = 0)

        u_test_ar = np.concatenate([np.copy(u_test), \
            add_lags(np.copy(u_test)[[f10_idx, kp_idx], :], lags = ar_lags)[2:, :]], axis = 0)

    z1_train_ar = add_lags(z1_train, lags = ar_lags)[n_components:, :]
    z2_train_ar = add_lags(z2_train, lags = ar_lags)[n_components:, :]

    if ar_state_flag is False:
        print('Not AR states in the run.')

    if u_train_ar.shape[1] == z1_train_ar.shape[1]:
        u_train_ar = np.concatenate([u_train_ar, 
            z1_train_ar], axis = 0)
    else: 
        u_train_ar = np.concatenate([u_train_ar, 
            np.zeros((int(n_components*max(ar_lags)), u_train_ar.shape[1]))], axis = 0)        

    return u_train_ar, z1_train_ar, z2_train_ar, u_test_ar

def interpolate_matrix_rows(matrix, time_conversion_ratio):
    #time_conversion_ratio is the number of new intervals between the current 
    n, m = matrix.shape
    converted_length = int(m * time_conversion_ratio)
    result = np.zeros((n, converted_length))
    
    for i in range(n):
        t_original = np.arange(m)
        t_interpolated = np.linspace(0, m - 1, converted_length)
        
        result[i, :] = scipy.interpolate.CubicSpline(t_original, matrix[i, :])(t_interpolated)
    
    return result

def interpolate_matrix_rows_linear_fast(matrix: np.ndarray, time_conversion_ratio: float) -> np.ndarray:
    """
    Linear interpolation along axis=1 (columns) for each row.

    time_conversion_ratio: factor for new number of samples.
      new_m = round(m * time_conversion_ratio)
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D (n, m)")

    n, m = matrix.shape
    new_m = int(np.round(m * time_conversion_ratio))
    if new_m <= 1 or m <= 1:
        # nothing meaningful to interpolate
        return matrix[:, :max(new_m, 0)].copy()

    # New sample positions in "old index" coordinates: [0, m-1]
    x_new = np.linspace(0.0, m - 1.0, new_m, dtype=np.float64)

    # Neighbor indices
    x0 = np.floor(x_new).astype(np.int64)
    x1 = np.minimum(x0 + 1, m - 1)

    # Linear weights
    w = (x_new - x0).astype(matrix.dtype, copy=False)  # shape (new_m,)

    # Gather and blend (broadcast over rows)
    y0 = matrix[:, x0]          # (n, new_m)
    y1 = matrix[:, x1]          # (n, new_m)
    return y0 + (y1 - y0) * w   # w broadcasts over axis 0

def discrete_to_continuous(A_d, B_d, delta_t):
    A_c = (1 / delta_t) * logm(A_d)
    B_c = A_c @ (np.linalg.inv(A_d - np.eye(A_d.shape[0])) @ B_d)
    return A_c, B_c

def change_resolution(A_d1, B_d1, T1, T2):
    A_d2 = scipy.linalg.fractional_matrix_power(A_d1, T2 / T1)
    A_d2 = np.real_if_close(A_d2, tol=1000)
    B_d2 = (A_d2 - np.eye(A_d2.shape[0])) @ np.linalg.inv(A_d1 - np.eye(A_d1.shape[0])) @ B_d1
    return A_d2, B_d2

def move_column(array, from_col, to_col):
    return np.insert(np.delete(array, from_col, axis=1), to_col, array[:, from_col], axis=1)

def normalize_array(array, method = 'std'):

    x = np.copy(array)

    if method == 'mean':
        x_mean = np.mean(x, axis = 1, keepdims = True)
        x_norm = (x - x_mean)
        norm_dict = {'x_mean': x_mean}  

    elif method == 'std':
        x_mean = np.mean(x, axis = 1, keepdims = True)
        x_std = np.std(x, axis = 1, keepdims = True)
        x_norm = (x - x_mean) / x_std
        norm_dict = {'x_mean': x_mean, 'x_std': x_std}

    elif method == 'minmax':
        max_values = np.max(x, axis = 1, keepdims = True)
        min_values = np.min(x, axis = 1, keepdims = True)
        x_norm = (x - min_values) / (max_values - min_values)
        norm_dict = {'x_min': min_values, 'x_max': max_values}

    elif method == 'none':
        x_norm = x
        norm_dict = {}

    return x_norm, norm_dict

def normalize_with_dict(x, norm_dict, method):
    x_temp = np.copy(x)
    if method == 'mean':
        x_norm = x_temp - norm_dict['x_mean']
    elif method == 'std':
        x_norm = (x_temp - norm_dict['x_mean']) / norm_dict['x_std']
    elif method == 'minmax':
        x_norm = (x_temp - norm_dict['x_min']) / (norm_dict['x_max'] - norm_dict['x_min'])
    elif method == 'none':
        x_norm = x_temp
    return x_norm

def denormalize(x, norm_dict, method):
    x_temp = np.copy(x)
    if method == 'mean':
        x_orig = x_temp + norm_dict['x_mean']
    elif method == 'std':
        x_orig = x_temp * norm_dict['x_std'] + norm_dict['x_mean']
    elif method == 'minmax':
        x_orig = x_temp * (norm_dict['x_max'] - norm_dict['x_min']) + norm_dict['x_min']
    elif method == 'none':
        x_orig = x_temp
    return x_orig

def build_regression_inputs(Y, z1, all_drivers, normalization_method, model_params, f10_idx, mdl = 'sindy', pca_couplings = 2, all_pca = False):
    kp_idx = f10_idx + 1
    f10a_idx = f10_idx + 2
    drivers = all_drivers[:(kp_idx + 1), :]
    print(drivers.shape)
    ar_drivers = all_drivers[(kp_idx + 1):, :]
    print(ar_drivers.shape)

    if mdl == 'sindy':
        Y_train_norm, Y_reg_norm_dict = normalize_array(Y, method = normalization_method) 
        Y_reg_norm_dict_sindy = Y_reg_norm_dict
        
        if all_pca == False:
            X_for_sindy_train = np.copy(np.concatenate([z1, drivers[f10_idx:, :]], axis = 0)) 
        else:
            X_for_sindy_train = np.copy(np.concatenate([z1, drivers[f10_idx:, :]], axis = 0)) 

        n_x = X_for_sindy_train.shape[0]
        print('Minmax before pre sindy normalization')
        print(X_for_sindy_train.max(), X_for_sindy_train.min())

        X_library_matrix_inputs_norm_dict = None
           
        if all_pca == False:
            input_features = [ f'x_{k}' for k in range(z1.shape[0])] + ['f107', 'kp'] # + ['t1', 't2', 't3', 't4', 'f107a', 'f107', 'kp']
        else:
            input_features = [ f'x_{k}' for k in range(z1.shape[0])] + ['f107', 'kp'] # + ['t1', 't2', 't3', 't4', 'f107a', 'f107', 'kp']
        
        library_dict = create_library_functions(np.copy(X_for_sindy_train.reshape((n_x, -1)).T), model_params['functions'], input_features)
        theta_train, theta_features = library_dict['theta'].T, library_dict['library_feature_names']
        print("Theta features ", theta_features)
        if all_pca == False:
            X_train = np.copy(np.concatenate([theta_train, ar_drivers, z1], axis = 0))
        else:
            X_train = np.copy(np.concatenate([theta_train, ar_drivers, z1], axis = 0))

        X_train = np.concatenate([np.copy(X_train[0, :]).reshape((1, -1)), np.copy(X_train[11:, :])], axis = 0)
        #np.copy(X_train[11:, :]) since X_train includes the first row which is constant

        X_train_norm, X_reg_norm_dict = normalize_array(X_train[1:, :], method = 'std')##### <------
        X_reg_norm_dict_sindy = X_reg_norm_dict
        X_train_norm = np.concatenate([X_train[0, :].reshape((1, -1)), drivers[1:f10_idx, :], X_train_norm])
        # print(X_train_norm[:, :10])
        print(X_train_norm.max(), X_train_norm.min())
        return X_train_norm, X_reg_norm_dict_sindy, Y_train_norm, Y_reg_norm_dict_sindy, \
            theta_features, X_library_matrix_inputs_norm_dict
    
    elif mdl == 'dmd':
        Y_train_norm, Y_reg_norm_dict = normalize_array(Y, method = normalization_method) 
        Y_reg_norm_dict_dmd = Y_reg_norm_dict
        X_train = np.copy(np.concatenate([z1, drivers[f10_idx:, :]], axis = 0)) 
        
        X_train_norm, X_reg_norm_dict = normalize_array(X_train, method = normalization_method)   
        X_train_norm = np.concatenate([drivers[1:f10_idx, :], X_train_norm]) 
        X_reg_norm_dict_dmd = X_reg_norm_dict     
        return X_train_norm, X_reg_norm_dict_dmd, Y_train_norm, Y_reg_norm_dict_dmd
    
    elif mdl == 'dmd-c':
        Y_train_norm, Y_reg_norm_dict = normalize_array(Y, method = normalization_method) 
        Y_reg_norm_dict_dmd = Y_reg_norm_dict
        X_train = np.copy(np.concatenate([drivers[f10_idx:, :], (drivers[f10_idx, :] * drivers[kp_idx, :]).reshape((1, -1)), 
            (drivers[kp_idx, :] * drivers[kp_idx, :]).reshape((1, -1)), z1], axis = 0))
        X_train_norm, X_reg_norm_dict = normalize_array(X_train, method = normalization_method)
        X_train_norm = np.concatenate([drivers[1:f10_idx, :], X_train_norm]) 
        X_reg_norm_dict_dmd = X_reg_norm_dict     
        return X_train_norm, X_reg_norm_dict_dmd, Y_train_norm, Y_reg_norm_dict_dmd
    
    elif mdl == 'nl-ddm':
        Y_train_norm, Y_reg_norm_dict = normalize_array(Y, method = normalization_method) 
        Y_reg_norm_dict_dmd = Y_reg_norm_dict
        X_train = np.copy(np.concatenate([drivers[f10_idx:, :], (drivers[f10_idx, :] * drivers[kp_idx, :]).reshape((1, -1)), 
            (drivers[kp_idx, :] * drivers[kp_idx, :]).reshape((1, -1)), z1], axis = 0))
        X_train_norm, X_reg_norm_dict = normalize_array(X_train, method = normalization_method)
        X_train_norm = np.concatenate([drivers[1:f10_idx, :], X_train_norm]) 
        X_reg_norm_dict_dmd = X_reg_norm_dict     
        return X_train_norm, X_reg_norm_dict_dmd, Y_train_norm, Y_reg_norm_dict_dmd


def import_satellite_data(satellite_name):
    columns = [
        'date', 'time', 'GPS', 'alt', 'lon', 'lat', 'lst', 'arglat',
        'accelerometer_density', 'dens_mean', 'flag_dens', 'flag_dens_mean'
    ]

    file_paths = sorted(glob.glob(f'./local_archive/{satellite_name}/*.txt'))
    df_list = []

    for file_path in file_paths:
        df = pd.read_csv(
            file_path,
            sep=r'\s+',
            comment='#',
            names=columns
        )
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df.set_index('datetime', inplace=True)
        df.drop(columns=['date', 'time', 'GPS'], inplace=True)
        
        df_list.append(df)

    champ_all = pd.concat(df_list)
    champ_all.sort_index(inplace=True)
    champ_hourly = champ_all.loc[    (champ_all.index.minute == 0) &
        (champ_all.index.second == 0)].copy()

    return champ_all, champ_hourly

def mat_to_lambda_dict(obj):
    temp_dict = {}

    for basis_code in obj._fieldnames:
        if basis_code.count('p') >= 1:
            powers = re.findall(r'(?:\d+\.\d+|\d+|\.\d+)', basis_code)

            if len(powers) != 1:
                raise ValueError(f"Ambiguous power in {basis_code}: {powers}")

            power = float(powers[0])

            # bind power at definition time
            temp_dict[basis_code] = (lambda x, p=power: x**p)

    return temp_dict


def create_library_functions(x, functions_dictionary, feature_names):
    function_names = functions_dictionary.keys()

    feature_names_lst = []
    for tuple in list(product(functions_dictionary, feature_names)):
        feature_names_lst.append(tuple[0] + '(' + tuple[1] + ')')

    feature_names_interactions_lst = []
    for tuple in list(combinations_with_replacement(feature_names_lst, 2)):
        feature_names_interactions_lst.append(tuple[0] + tuple[1] )

    all_feature_names = ['1'] + feature_names_lst + feature_names_interactions_lst 

    arr = np.concatenate([np.apply_along_axis(functions_dictionary[func_name], 0, x) for func_name in function_names], axis = 1)
    combs = list(combinations_with_replacement(range(arr.shape[1]), 2))
    
    left_combinations_index = [p[0] for p in combs]
    right_combinations_index = [p[1] for p in combs]
    
    theta = np.concatenate([np.ones(arr.shape[0]).reshape((-1, 1)), arr, arr[:, left_combinations_index] * arr[:, right_combinations_index]], axis = 1)
    library_dict = {'theta': theta, 'library_feature_names': all_feature_names}
    return library_dict



def create_library_functions_optimized(x, functions_dictionary, feature_names, return_names=True):

    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    func_names = list(functions_dictionary.keys())
    n_samples, n_feat = x.shape
    n_funcs = len(func_names)

    if feature_names is None:
        feature_names_fixed = [f"x{i}" for i in range(n_feat)]
    else:
        feature_names = list(feature_names)
        if len(feature_names) >= n_feat:
            feature_names_fixed = feature_names[:n_feat]
        else:
            feature_names_fixed = feature_names + [f"x{i}" for i in range(len(feature_names), n_feat)]

    unary_blocks = []
    for fn in func_names:
        y = functions_dictionary[fn](x)
        y = np.asarray(y)

        if y.shape != x.shape:
            if y.shape == (n_samples,):
                y = y.reshape(n_samples, 1).repeat(n_feat, axis=1)
            elif y.shape == (n_samples, 1):
                y = y.repeat(n_feat, axis=1)
            else:
                raise ValueError(f"Function '{fn}' returned shape {y.shape}, expected {x.shape} (or broadcastable).")

        unary_blocks.append(y)

    arr = np.concatenate(unary_blocks, axis=1)   
    m = arr.shape[1]                             

    iu, ju = np.triu_indices(m)
    n_int = iu.size

    theta = np.empty((n_samples, 1 + m + n_int), dtype=arr.dtype)
    theta[:, 0] = 1.0
    theta[:, 1:1+m] = arr
    theta[:, 1+m:] = arr[:, iu] * arr[:, ju]

    if not return_names:
        return {"theta": theta}

    unary_names = [f"{fn}({feat})" for fn in func_names for feat in feature_names_fixed]
    if len(unary_names) != m:
        unary_names = [f"u{i}" for i in range(m)]

    interaction_names = [unary_names[i] + unary_names[j] for i, j in zip(iu, ju)]
    all_feature_names = ["1"] + unary_names + interaction_names

    return {"theta": theta, "library_feature_names": all_feature_names}



def find_closest_match(df, f10_target, kp_target):
    df['f10_diff'] = np.abs(df['f10'] - f10_target)
    df['kp_diff'] = np.abs(df['kp'] - kp_target)
    
    closest_row = df.loc[(df['f10_diff'] + df['kp_diff']).idxmin()]
    closest_row = closest_row.drop(columns=['f10_diff', 'kp_diff'])
    df = df.drop(columns=['f10_diff', 'kp_diff'])
    
    return closest_row


def get_propagator_datetimes(datetimes_ds):
    if isinstance(datetimes_ds, np.ndarray) and np.issubdtype(datetimes_ds.dtype, np.number):
        datetimes_ds = matlab_datenum_to_py_datetime(datetimes_ds)
        return datetimes_ds
    elif np.isscalar(datetimes_ds) and isinstance(datetimes_ds, (float, int)):
        datetimes_ds = matlab_datenum_to_py_datetime([datetimes_ds])
        return datetimes_ds

def build_dyn_sindy_inputs(z1_k_denormalized, all_drivers, X_reg_norm_dict_sindy, f10_idx, model_params,\
        normalization_method, input_features, time):
    kp_idx = f10_idx + 1
    f10a_idx = kp_idx + 1
    drivers = np.copy(all_drivers[:(kp_idx + 1), time])
    ar_drivers = np.copy(all_drivers[(kp_idx + 1):, time])
    X_k_for_sindy = np.concatenate([z1_k_denormalized.reshape((-1, 1)), drivers[f10_idx:].reshape((-1, 1))])

    library_dict = create_library_functions(np.copy(X_k_for_sindy.T), model_params['functions'], input_features)
    theta_k = library_dict['theta'].T
    X_k = np.concatenate([np.ones((1, 1)), theta_k[11:, 0].reshape((-1, 1)), ar_drivers.reshape((-1, 1)), z1_k_denormalized], axis = 0)
    
    X_k_norm = normalize_with_dict(X_k[1:], X_reg_norm_dict_sindy, method = normalization_method)
    X_k_norm = np.concatenate([X_k[0, 0].reshape((1, 1)), drivers[1:f10_idx].reshape((-1, 1)), X_k_norm], axis = 0) #adding the constant column to the inputs

    return X_k_norm

def sindy_loop(u_prop_interpolated, X_reg_norm_dict, Y_reg_norm_dict, A_d2, B_ar_drivers_lst, B_ar_state_lst, B_nonlinear, \
    stored_frcst, ar_drivers_start, t0, tf, ar_lags, time_conversion_factor, f10_idx, normalization_method, model_params, input_features):
    kp_idx = f10_idx + 1
    delta_idx = 5
    X_ar_drivers_norm_dict = {'x_mean': X_reg_norm_dict['x_mean'][(ar_drivers_start - delta_idx):(ar_drivers_start + 2*max(ar_lags) - delta_idx), :].reshape((-1, 1)), \
        'x_std': X_reg_norm_dict['x_std'][(ar_drivers_start - delta_idx):(ar_drivers_start + 2*max(ar_lags) - delta_idx), :].reshape((-1, 1))}

    for k in range(int(t0), int(tf)):

        #Getting the different input variables blocks
        #stored_frcst goes into the future as the index advances through columns
        z_ar_np_interpolated = np.fliplr(stored_frcst[:, -int((max(ar_lags) + 1) * time_conversion_factor):-int(time_conversion_factor)])
    
        #Latent space componbents need to be interpolated only once before the beginning of the loop, as well
        #as with AR drtivers. Then, SINDY variables can be constructed within the loop each time. 
        #AR drivers can be interpolated once, and the picked at each iteration.
        #SINDY inputs need to be built from scratch and then be normalized.
        #Calculation of drivers and AR dtrivers variables.
        L = len(ar_lags)
        ar_drivers_prop_interpolated = u_prop_interpolated[( kp_idx + 1 ) : ( ( kp_idx + 1 ) + 2 * L ), k].reshape((-1, 1))#This goes into the past  

        ar_drivers_prop_interpolated = normalize_with_dict(np.copy(ar_drivers_prop_interpolated), \
            X_ar_drivers_norm_dict, method = normalization_method)
        # ar_drivers_prop_interpolated = ar_drivers_prop_interpolated * ar_drivers_weights

        ar_drivers_prop_interpolated = ar_drivers_prop_interpolated.reshape(L, 2).T

        z_k_norm = np.copy(stored_frcst[:, -1]).reshape((-1, 1))

        z_k_denormalized = denormalize(np.copy(z_k_norm), norm_dict=Y_reg_norm_dict, method=normalization_method)
        sindy_inputs_norm = build_dyn_sindy_inputs(z_k_denormalized, u_prop_interpolated, \
            X_reg_norm_dict, f10_idx, model_params, \
                normalization_method, input_features, time = k)
        #SINDY variables ()
        x_nonlinear = np.copy(sindy_inputs_norm[:ar_drivers_start].reshape((-1, 1)))

        #Forecasting next step
        ar_state_contribution = np.sum([B_ar_state_lst[i] @ z_ar_np_interpolated[:, i].reshape((-1, 1)) for i in range(len(ar_lags))], axis = 0)       
        ar_drivers_contribution = np.sum([B_ar_drivers_lst[i] @ ar_drivers_prop_interpolated[:, i].reshape((-1, 1)) for i in range(len(ar_lags))], axis = 0)    

        z_frcst = A_d2 @ z_k_norm + ar_state_contribution + ar_drivers_contribution + B_nonlinear @ x_nonlinear
        # if max(z_frcst) >= 10.:
        #     stored_frcst = np.concatenate([stored_frcst, z_frcst], axis = 1)
        #     return stored_frcst
        stored_frcst = np.concatenate([stored_frcst, z_frcst], axis = 1)

    return stored_frcst



def build_dyn_sindy_inputs_optimized(
    z1_k_denormalized,
    all_drivers,
    X_reg_norm_dict_sindy,
    f10_idx,
    model_params,
    normalization_method,
    input_features,
    time,
    ):
    kp_idx = f10_idx + 1

    dcol = all_drivers[:, time]                       
    drivers = dcol[:kp_idx + 1]                       
    ar_drivers = dcol[kp_idx + 1:]                    

    zcol = z1_k_denormalized.reshape(-1, 1)           
    drv_f10kp = drivers[f10_idx:].reshape(-1, 1)      

    n_z = zcol.shape[0]
    n_f = drv_f10kp.shape[0]
    X_k_for_sindy = np.empty((n_z + n_f, 1), dtype=zcol.dtype)
    X_k_for_sindy[:n_z, 0] = zcol[:, 0]
    X_k_for_sindy[n_z:, 0] = drv_f10kp[:, 0]

    library_dict = create_library_functions_optimized(
        X_k_for_sindy.T, model_params["functions"], input_features, return_names = False
    )
    theta = library_dict["theta"] 

    theta_row = theta[0] if theta.ndim == 2 else theta
    theta_tail = np.asarray(theta_row[11:], dtype=X_k_for_sindy.dtype).reshape(-1, 1)

    arcol = ar_drivers.reshape(-1, 1)

    n_theta_tail = theta_tail.shape[0]
    n_ar = arcol.shape[0]
    n_x = 1 + n_theta_tail + n_ar + n_z

    X_k = np.empty((n_x, 1), dtype=X_k_for_sindy.dtype)
    i = 0
    X_k[i, 0] = 1.0
    i += 1

    X_k[i:i+n_theta_tail, 0] = theta_tail[:, 0]
    i += n_theta_tail

    X_k[i:i+n_ar, 0] = arcol[:, 0]
    i += n_ar

    X_k[i:i+n_z, 0] = zcol[:, 0]

    X_k_rest_norm = normalize_with_dict(X_k[1:], X_reg_norm_dict_sindy, method=normalization_method)

    mid = drivers[1:f10_idx].reshape(-1, 1)

    out = np.empty((1 + mid.shape[0] + X_k_rest_norm.shape[0], 1), dtype=X_k_for_sindy.dtype)
    j = 0
    out[j, 0] = X_k[0, 0]
    j += 1

    if mid.size:
        out[j:j+mid.shape[0], 0] = mid[:, 0]
        j += mid.shape[0]

    out[j:, 0] = X_k_rest_norm[:, 0]

    return out


def sindy_loop_optimized(
    u_prop_interpolated,
    X_reg_norm_dict,
    Y_reg_norm_dict,
    A_d2,
    B_ar_drivers_lst,
    B_ar_state_lst,
    B_nonlinear,
    stored_frcst,
    ar_drivers_start,
    t0,
    tf,
    ar_lags,
    time_conversion_factor,
    f10_idx,
    normalization_method,
    model_params,
    input_features,
):

    kp_idx = f10_idx + 1
    delta_idx = 5

    ar_lags = np.asarray(ar_lags, dtype=int)
    L = ar_lags.size
    tc = int(time_conversion_factor)
    n_state, n_init = stored_frcst.shape
    n_steps = int(tf) - int(t0)
    if n_steps <= 0:
        return stored_frcst

    sl0 = ar_drivers_start - delta_idx
    sl1 = ar_drivers_start + 2 * int(ar_lags.max()) - delta_idx
    X_ar_drivers_norm_dict = {
        "x_mean": X_reg_norm_dict["x_mean"][sl0:sl1, :].reshape((-1, 1)),
        "x_std":  X_reg_norm_dict["x_std"][sl0:sl1, :].reshape((-1, 1)),
    }

    B_state = np.stack(B_ar_state_lst, axis=0)      
    
    B_drv   = np.stack(B_ar_drivers_lst, axis=0)

    out = np.empty((n_state, n_init + n_steps), dtype=stored_frcst.dtype)
    out[:, :n_init] = stored_frcst

    lag_offsets = ar_lags * tc 

    cur = n_init - 1

    for k in range(int(t0), int(tf)):
        z_k_norm = out[:, cur:cur+1]

        cols = cur - lag_offsets
        z_ar = out[:, cols]

        ar_drv = u_prop_interpolated[(kp_idx + 1):(kp_idx + 1 + 2 * L), k].reshape((-1, 1))
        ar_drv = normalize_with_dict(ar_drv, X_ar_drivers_norm_dict, method=normalization_method)
        ar_drv = ar_drv.reshape(L, 2).T

        z_k_denorm = denormalize(z_k_norm, norm_dict=Y_reg_norm_dict, method=normalization_method)
        sindy_inputs_norm = build_dyn_sindy_inputs_optimized(
            z_k_denorm,
            u_prop_interpolated,
            X_reg_norm_dict,
            f10_idx,
            model_params,
            normalization_method,
            input_features,
            time=k,
        )

        x_nonlinear = sindy_inputs_norm[:ar_drivers_start].reshape((-1, 1))

        ar_state_contribution = np.einsum("lij,jl->i", B_state, z_ar).reshape((-1, 1))

        ar_drivers_contribution = np.einsum("lik,kl->i", B_drv, ar_drv).reshape((-1, 1))

        z_next = (A_d2 @ z_k_norm
                  + ar_state_contribution
                  + ar_drivers_contribution
                  + (B_nonlinear @ x_nonlinear))

        out[:, cur + 1:cur + 2] = z_next
        cur += 1

    return out




def dmd_loop(u_ds_interpolated, A_d2, B_nonlinear, stored_frcst, t0, tf, X_reg_norm_dict, \
        Y_reg_norm_dict, f10_idx, kp_idx, normalization_method, n_components):
    x_nonlinear = np.concatenate([u_ds_interpolated[f10_idx:, :], \
        (u_ds_interpolated[f10_idx, :] * u_ds_interpolated[kp_idx, :]).reshape((1, -1)), 
            (u_ds_interpolated[kp_idx, :] * u_ds_interpolated[kp_idx, :]).reshape((1, -1))], axis = 0)
    for k in range(int(t0), int(tf)):
        #Getting the different input variables blocks
        z_k = stored_frcst[:, -1].reshape((-1, 1))
        x_nonlinear_k = np.concatenate([np.copy(x_nonlinear[:, k]).reshape((-1, 1)), z_k.reshape((-1, 1))], axis = 0)

        x_nonlinear_norm = normalize_with_dict(x_nonlinear_k, X_reg_norm_dict, normalization_method)
        x_nonlinear_norm = np.concatenate([np.copy(u_ds_interpolated[1:f10_idx, k]).reshape((-1, 1)), x_nonlinear_norm], axis = 0) 

        z_frcst = A_d2 @ z_k.reshape((-1, 1)) + B_nonlinear @ x_nonlinear_norm[:-n_components].reshape((-1, 1))

        stored_frcst = np.concatenate([stored_frcst, z_frcst], axis = 1)

    return stored_frcst

def propagate(start_date, stop_date, model_object, ar_lags, T1, T2, time_conversion_factor, n_components, f10_idx = 5, \
        model_name = 'sindy-c', B_sindy = None, drivers = 'tiegcm2'):
    kp_idx = f10_idx + 1
    if model_name == 'sindy-c':
        B = model_object['B_sindy']
        X_reg_norm_dict = {'x_mean': np.copy(model_object['X_sindy_norm_dict'].x_mean).reshape((-1, 1)), 'x_std': np.copy(model_object['X_sindy_norm_dict'].x_std.reshape((-1, 1)))}
        Y_reg_norm_dict = {'x_mean': np.copy(model_object['Y_sindy_norm_dict'].x_mean).reshape((-1, 1)), 'x_std': np.copy(model_object['Y_sindy_norm_dict'].x_std.reshape((-1, 1)))}
    elif model_name == 'dmd-c':
        B = model_object['B_dmd']
        u_ar_ds = model_object['u_ar_ds']
        X_reg_norm_dict = {'x_mean': np.copy(model_object['X_dmd_ds_norm_dict'].x_mean).reshape((-1, 1)), 'x_std': np.copy(model_object['X_dmd_ds_norm_dict'].x_std.reshape((-1, 1)))}
        Y_reg_norm_dict = {'x_mean': np.copy(model_object['Y_dmd_ds_norm_dict'].x_mean).reshape((-1, 1)), 'x_std': np.copy(model_object['Y_dmd_ds_norm_dict'].x_std.reshape((-1, 1)))}
    elif model_name == 'nl-ddm':
        B = model_object['B_nl_ddm']
        u_ar_ds = model_object['X_nl_ddm_ds_norm']
        Y_reg_norm_dict = {'x_mean': np.copy(model_object['Y_nl_ddm_ds_norm_dict'].x_mean).reshape((-1, 1)), 'x_std': np.copy(model_object['Y_nl_ddm_ds_norm_dict'].x_std.reshape((-1, 1)))}
    else:
        raise ValueError("Invalid model name")
    if drivers == 'tiegcm2':
        u_ar_ds = model_object['u_ar_ds_tiegcm2']
        datetimes_ds = model_object['datetimes_ds_tiegcm2']
    else:
        u_ar_ds = model_object['u_ar_ds']
        datetimes_ds = model_object['datetimes_ds']
    normalization_method = model_object['normalization_method']
    model_params_mat = model_object['model_params']

    model_params = {'normalization_method':model_params_mat.normalization_method, 'shift':model_params_mat.shift, 'dof':model_params_mat.dof, 'functions':mat_to_lambda_dict(model_params_mat.functions), 'model':model_params_mat.model, 'control':model_params_mat.control}
    input_features = model_object['input_features']
    
    # z_ds = model_object["z_ds"]
    ic_df = pd.DataFrame(model_object['ic_array'], columns = ['f10', 'kp'] + [f'z_{str(k+1).zfill(2)}' for k in range(10)])
    datetime_series = pd.date_range(start=start_date, end = stop_date, freq = f'{int(T2)}s')

    datetimes_ds = get_propagator_datetimes(datetimes_ds)
    propagation_idx = np.array([(start_date <= date_time <=stop_date) for date_time in datetimes_ds])
    
    n_hours = (stop_date - start_date).total_seconds()/3600
    t0 = 0 * time_conversion_factor
    tf = np.floor((n_hours)) * time_conversion_factor
    # print(z_ds.shape, propagation_idx.shape)
    # z_ds_prop = np.copy(z_ds[:, propagation_idx])
    u_ar_ds_prop = np.copy(u_ar_ds[:, propagation_idx])
    u_ds_prop = np.copy(u_ar_ds_prop[ : ( kp_idx + 1 ), :])


    if model_name == 'sindy-c':
        if B_sindy is not None:
            B = B_sindy
        A_d = np.copy(B[:, -n_components:])
        B_d = np.copy(B[:, :-n_components])

        A_d2, B_d2 = change_resolution(A_d, B_d, T1, T2)

        ar_drivers_start = B.shape[1] - n_components - max(ar_lags) * n_components - 2 * max(ar_lags)
        ar_state_start = B.shape[1] - n_components - len(ar_lags)*n_components

        B_nonlinear = np.copy(B_d2[:, :ar_drivers_start])
        B_ar_drivers_lst = [np.copy(B_d2[:, slice(ar_drivers_start + k * 2, ar_drivers_start + (k + 1) * 2)]) for k in range(len(ar_lags))]#The more idx advances, the more goes into the past
        B_ar_state_lst = [np.copy(B_d2[:, slice(ar_state_start + k * n_components, ar_state_start + (k + 1) * n_components)]) for k in range(len(ar_lags))]#The more idx advances, the more goes into the past


        ti_lst = ar_lags
        # z_i_lst = [np.copy(z_ds_prop[:, k].reshape((-1, 1))) for k in range(max(ti_lst) + 1)]#this is going into the future as k increases
        z_i_lst = [find_closest_match(ic_df, u_ds_prop[f10_idx, k], u_ds_prop[kp_idx, k]).values[2:12].reshape((-1, 1)) for k in range(max(ti_lst) + 1)]
        # print(z_i_lst)
        # print(z_i_lst2)
        # aaa
        x_drivers_prop = np.copy(u_ar_ds_prop)
        x_drivers_prop_interpolated = interpolate_matrix_rows(x_drivers_prop, time_conversion_factor)
        stored_frcst = np.concatenate(z_i_lst, axis = 1)#This is going into the future as the column index increases
        stored_frcst = normalize_with_dict(stored_frcst, Y_reg_norm_dict, method = normalization_method)
        stored_frcst = interpolate_matrix_rows(stored_frcst, time_conversion_factor)
        
        stored_frcst = sindy_loop_optimized(x_drivers_prop_interpolated, X_reg_norm_dict, Y_reg_norm_dict, A_d2, B_ar_drivers_lst, \
            B_ar_state_lst, B_nonlinear, stored_frcst, ar_drivers_start, t0, tf, ar_lags, \
                time_conversion_factor, f10_idx, normalization_method, model_params, input_features)[:, int((max(ar_lags) - 2) * time_conversion_factor):-2]

        z_frcst_np = denormalize(stored_frcst, norm_dict=Y_reg_norm_dict, method=normalization_method) #np.copy(stored_frcst) * Y_reg_norm_dict['x_std'] + Y_reg_norm_dict['x_mean']
        z_frcst_np_hourly = np.copy(stored_frcst[:, ::int(time_conversion_factor)]) * Y_reg_norm_dict['x_std'] + Y_reg_norm_dict['x_mean']
        return z_frcst_np[:, :datetime_series.shape[0]], z_frcst_np_hourly, x_drivers_prop[:(kp_idx+1), :], datetime_series

    elif model_name == 'dmd-c':
        A_d = np.copy(B[:, -n_components:])
        B_d = np.copy(B[:, :-n_components])

        A_d2, B_d2 = change_resolution(A_d, B_d, T1, T2)

        ti_lst = [ar_lags[0]]
        # z_i_lst = [np.copy(z_ds_prop[:, k].reshape((-1, 1))) for k in ti_lst]#this is going into the future as k increases
        z_i_lst = [find_closest_match(ic_df, u_ds_prop[f10_idx, k], u_ds_prop[kp_idx, k]).values[2:12].reshape((-1, 1)) for k in ti_lst]
        # print(z_i_lst)
        # print(z_i_lst2)
        # aaa
        u_ds_prop_interpolated = interpolate_matrix_rows(u_ds_prop, time_conversion_factor)
        stored_frcst = np.array(z_i_lst).reshape((-1, 1))
        stored_frcst = normalize_with_dict(stored_frcst, Y_reg_norm_dict, method = normalization_method)
        stored_frcst = dmd_loop(u_ds_prop_interpolated, A_d2, B_d2, stored_frcst, t0, tf, X_reg_norm_dict, \
            Y_reg_norm_dict, f10_idx, kp_idx, normalization_method, n_components)

        z_frcst_np = np.copy(stored_frcst) * Y_reg_norm_dict['x_std'] + Y_reg_norm_dict['x_mean']
        z_frcst_np_hourly = np.copy(stored_frcst[:, ::int(time_conversion_factor)]) * Y_reg_norm_dict['x_std'] + Y_reg_norm_dict['x_mean']
        return z_frcst_np[:, :datetime_series.shape[0]], z_frcst_np_hourly, u_ds_prop, datetime_series

    elif model_name == 'nl-ddm':
        A_d = np.copy(B[:, -n_components:])
        B_d = np.copy(B[:, :-n_components])

        A_d2, B_d2 = change_resolution(A_d, B_d, T1, T2)

        ti_lst = [ar_lags[0]]
        z_i_lst = [np.copy(u_ar_ds_prop[-n_components:, k].reshape((-1, 1))) for k in ti_lst]#this is going into the future as k increases
        x_prop_norm_interpolated = interpolate_matrix_rows(u_ar_ds_prop, time_conversion_factor)

        stored_frcst = np.array(z_i_lst).reshape((-1, 1))
        stored_frcst = dmd_loop(x_prop_norm_interpolated, A_d2, B_d2, stored_frcst, t0, tf, n_components)
        
        z_frcst_np = denormalize(stored_frcst, norm_dict=Y_reg_norm_dict, method=normalization_method) #np.copy(stored_frcst) * Y_reg_norm_dict['x_std'] + Y_reg_norm_dict['x_mean']
        z_frcst_np_hourly = np.copy(stored_frcst[:, ::int(time_conversion_factor)]) * Y_reg_norm_dict['x_std'] + Y_reg_norm_dict['x_mean']
        return z_frcst_np[:, :datetime_series.shape[0]], z_frcst_np_hourly, x_prop_norm_interpolated[:(kp_idx+1), int((max(ar_lags) - 2) * time_conversion_factor):-2], datetime_series




def interpolate(output_datetimes: np.array, timestamps: np.array, lla: np.array, model_object: object, z_dict: dict) -> tuple[float, float]:
    warnings.filterwarnings("ignore")

    lt_axis = np.linspace(0, 23.66666667, 72)
    lat_axis = np.linspace(-87.5, 87.5, 36)
    alt_axis = np.linspace(100, 980, 45)  # km

    U0 = model_object['U'].reshape((72, 36, 45, 10), order='C')
    mu0 = model_object['mu'].reshape((72, 36, 45), order='C')

    timestamps = pd.to_datetime(timestamps)
    
    original_time_variations = (output_datetimes - output_datetimes[0]).total_seconds()
    time_variations = (timestamps - output_datetimes[0]).total_seconds()
    # print(np.max(T), t.min(), t.max())
    
    points = np.column_stack([lla[:, 1], lla[:, 0], lla[:, 2]])  # (LAT, LT, ALT) ---> (LT, LAT, ALT)

    # Prepare interpolation variables
    # Define grid for interpolation

    lt = np.linspace(lt_axis[0], lt_axis[-1], U0.shape[0])
    lat = np.linspace( lat_axis[0], lat_axis[-1], U0.shape[1]) 
    alt = np.linspace(alt_axis[0], alt_axis[-1], U0.shape[2])
    
    my_interpolating_U0 = rgi( (lt, lat, alt), U0, bounds_error=False, fill_value=None)
    my_interpolating_mu0 = rgi( (lt, lat, alt), mu0, bounds_error=False, fill_value=None) 
    
    # Perform interpolation
    U0_interp = my_interpolating_U0(points)
    mu0_interp = my_interpolating_mu0(points)

    sindy_interpolators_lst = []
    for sindy_model in list(z_dict.keys()):
        sindy_interpolators_lst.append(interp1d(original_time_variations, z_dict[sindy_model], \
            kind='linear', axis=1, fill_value = "extrapolate"))

    z_uncertainties_values_lst = [ 10 ** (np.sum(U0_interp * interpolator(time_variations).T, axis = 1).reshape((-1, 1)) + \
        mu0_interp.T.reshape((-1, 1))) for interpolator in sindy_interpolators_lst]
    interpolated_models = np.stack(z_uncertainties_values_lst).squeeze(-1).T

    density_std = np.nanstd(interpolated_models, axis = 1)
    density = np.nanmean(interpolated_models, axis = 1)

    return interpolated_models, density, density_std