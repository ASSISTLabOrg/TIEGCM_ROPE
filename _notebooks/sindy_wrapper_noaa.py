# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
from time import perf_counter
from sklearn.decomposition import TruncatedSVD
import utilities_ds as u
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.preprocessing import scale 
import matplotlib.pyplot as plt
import os 
import h5py
import pickle
import warnings
import itertools
from scipy.linalg import logm
from scipy.integrate import solve_ivp

# %%
upload_path = '.' #/users/cires/codes/sindy/data'
input_data_tiegcm = np.load(os.path.join(upload_path, 'bundle_rho_u_svd.npz'), allow_pickle=True)
input_data_models = np.load(os.path.join(upload_path, 'models_bundle.npz'), allow_pickle=True)

class SvdContainer:
    def __init__(self, u_svd, mu):
        self.U = u_svd
        self.mu = mu

rho_test = input_data_tiegcm['rho']
drivers_test = input_data_tiegcm['u']
u_svd = input_data_tiegcm['u_svd']
mu_svd = input_data_tiegcm['mu_svd']

X_reg_norm_dict_nl_dmd = input_data_models['X_reg_norm_dict_nl_dmd'][()]
Y_reg_norm_dict_nl_dmd = input_data_models['Y_reg_norm_dict_nl_dmd'][()]

x_train_svd_obj = SvdContainer(u_svd, mu_svd)
x_train_svd_obj.norm_dict = {'x_mean': mu_svd}
x_train_svd_obj.norm_dict.keys()

# %%

# %%
import sys

def get_size(obj, seen=None):
    """Recursively calculate the size of objects."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Mark the object as seen
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum(get_size(v, seen) for v in obj.values())
        size += sum(get_size(k, seen) for k in obj.keys())
    elif isinstance(obj, (list, set, tuple)):
        size += sum(get_size(i, seen) for i in obj)
    return size

def move_column(array, from_col, to_col):
    return np.insert(np.delete(array, from_col, axis=1), to_col, array[:, from_col], axis=1)

def build_sindy_dyn_frcst_inputs(z1_k, drivers, X_reg_norm_dict_sindy, pca_coupling, kp_idx, f10_idx, model_params, normalization_method, input_features, k = 0):
    X_k_for_sindy = np.concatenate([z1_k[pca_coupling].reshape((1, 1)), drivers[f10_idx:, k].reshape((-1, 1))])
    current_kp = drivers[kp_idx, k]
    library_dict = u.create_library_functions(np.copy(X_k_for_sindy.T), model_params['functions'], input_features)
    theta_k = library_dict['theta'].T
    X_k = np.concatenate([theta_k, np.delete(z1_k, pca_coupling, axis = 0)], axis = 0)     
    X_k_norm = u.normalize_with_dict(X_k[1:], X_reg_norm_dict_sindy, method = normalization_method)  
    X_k_norm = np.concatenate([X_k[0, :].reshape((1, -1)), drivers[1:f10_idx, k].reshape((-1, 1)), X_k_norm]) 

    return X_k_norm

# ODE function
def ode_func(t, q_norm, drivers, A_c, B_c, sindy_tgt_col):
    #########################Here comes the interpolation################################
    # discrete_idx = np.searchsorted(t_interval[::4], t) - 1
    # discrete_idx = max(0, min(int(discrete_idx), len(t_interval[::4]) - 1)) 
    ##################################################################################### 
    discrete_idx = np.searchsorted(t_interval, t)
    discrete_idx = max(0, min(int(discrete_idx), len(t_interval) - 1)) 
    
    q_denormalized = u.denormalize(q_norm.reshape((-1, 1)), Y_reg_norm_dict, normalization_method)
    X_k_norm = build_sindy_dyn_frcst_inputs(q_denormalized, drivers, X_reg_norm_dict_sindy, pca_coupling, kp_idx, f10_idx, model_params,\
        normalization_method, input_features, k = int(t)) 
    qF_norm = move_column(np.copy(X_k_norm).T, 5, sindy_tgt_col).T    
    F_norm = np.copy(qF_norm[:(-10), 0]).reshape((-1, 1))
    dq_dt = (A_c @ q_norm.reshape((-1, 1)) + B_c @ F_norm.reshape((-1, 1))).flatten()   
    return dq_dt

# ODE function
def ode_func_sindy(t, q_norm, drivers, A_sindy_joint_c, B_sindy_joint_c, A_sindy_f10_c, B_sindy_f10_c, sindy_tgt_col):
    #########################Here comes the interpolation################################
    # discrete_idx = np.searchsorted(t_interval[::4], t) - 1
    # discrete_idx = max(0, min(int(discrete_idx), len(t_interval[::4]) - 1)) 
    ##################################################################################### 
    discrete_idx = np.searchsorted(t_interval, t)
    discrete_idx = max(0, min(int(discrete_idx), len(t_interval) - 1)) 
    
    current_kp = drivers[kp_idx, int(t)]
    
    q_denormalized = u.denormalize(q_norm.reshape((-1, 1)), Y_reg_norm_dict, normalization_method)
    X_k_norm = build_sindy_dyn_frcst_inputs(q_denormalized, drivers, X_reg_norm_dict_sindy, pca_coupling, kp_idx, f10_idx, model_params,\
        normalization_method, input_features, k = int(t)) 
    qF_norm = move_column(np.copy(X_k_norm).T, 5, sindy_tgt_col).T    
    F_norm = np.copy(qF_norm[:(-10), 0]).reshape((-1, 1))
    dq_dt = ((A_sindy_joint_c * (current_kp >= 3) + A_sindy_f10_c * (current_kp < 3)) @ q_norm.reshape((-1, 1)) + (B_sindy_joint_c * (current_kp >= 3) + B_sindy_f10_c * (current_kp < 3)) @ F_norm.reshape((-1, 1))).flatten()   
    return dq_dt

def ode_func_dmd(t, q_norm, drivers, A_c, B_c):
    #########################Here comes the interpolation################################
    # discrete_idx = np.searchsorted(t_interval[::4], t) - 1
    # discrete_idx = max(0, min(int(discrete_idx), len(t_interval[::4]) - 1)) 
    ##################################################################################### 
    discrete_idx = np.searchsorted(t_interval, t)
    discrete_idx = max(0, min(int(discrete_idx), len(t_interval) - 1)) 
    k = int(t)
    
    q_denormalized = u.denormalize(q_norm.reshape((-1, 1)), Y_reg_norm_dict, normalization_method)
    
    X_k = np.concatenate([q_denormalized, drivers[f10_idx:, k].reshape((-1, 1)), (drivers[f10_idx, k] * drivers[kp_idx, k]).reshape((-1, 1)), 
        (drivers[kp_idx, k] * drivers[kp_idx, k]).reshape((-1, 1))])
    X_k_norm = u.normalize_with_dict(X_k, X_reg_norm_dict_nl_dmd, normalization_method)
    X_k_norm = np.concatenate([drivers[1:f10_idx, k].reshape((-1, 1)), X_k_norm])
    
    q0_norm = np.copy(X_k_norm[4:(-4), :])
    F_norm = np.copy(np.concatenate([X_k_norm[:4, :], X_k_norm[(-4):, :]], axis = 0))
    
    dq_dt = (A_c @ q_norm.reshape((-1, 1)) + B_c @ F_norm.reshape((-1, 1))).flatten()   
    return dq_dt

def interpolate_matrix_rows(matrix, sub_intervals):
    n, m = matrix.shape
    interpolated_columns = m * sub_intervals 
    result = np.zeros((n, interpolated_columns))
    
    for i in range(n):
        x_original = np.arange(m)
        x_interpolated = np.linspace(0, m - 1, interpolated_columns)
        
        result[i, :] = np.interp(x_interpolated, x_original, matrix[i, :])
    
    return result


# %%
input_data_models

# %%
dataset_dict_all = {}
shift = 1
normalization_method = 'std'
dof = x_train_svd_obj.U.shape[0]

dataset_dict_all['drivers_data'] = input_data_tiegcm['u']
dataset_dict_all['state_data'] = input_data_tiegcm['rho']
x_train_dict = input_data_tiegcm['svd_centering_dict'][()]
doy_all_train = input_data_tiegcm['doy_whole'][()]

models_dict = input_data_models['models_dict'][()]
ridge_parameters = input_data_models['ridge_parameters'][()]
bf_normalization_dict = input_data_models['bf_normalization_dict'][()]
X_reg_norm_dict_nl_dmd = input_data_models['X_reg_norm_dict_nl_dmd'][()]
Y_reg_norm_dict_nl_dmd = input_data_models['Y_reg_norm_dict_nl_dmd'][()]

model_params = {'normalization_method': normalization_method, 'shift': shift, 'dof': dof, 'functions': {},
                'model': 'dmd', 'control': True}


T = [15.34175, 12.2734, 24., 48., 20.45566667, 30.6835, 61.367]
poly1 = {'p1': lambda x: x}
poly2 = {'p2': lambda x: x**2}
poly3 = {'p3': lambda x: x**3}
exp1 = {'e1': lambda x: np.exp(-x)}
sincos4 = {'g13': lambda x: np.sin(2*np.pi*x/T[3]), 'g14': lambda x: np.cos(2*np.pi*x/T[3])}
sincos7 = {'g13': lambda x: np.sin(2*np.pi*x/T[6]), 'g14': lambda x: np.cos(2*np.pi*x/T[6])}

basis_functions_dict = {'poly':poly1}

# %%

# %%
centering_method = 'mean'
segment = 'all'
n_days = 5
f10_idx = 5
kp_idx = 6
n_components = 10

segments = ['quiet', 'moderate', 'strong', 'all']
normalization_method = 'std'

days_backward = 2
days_forward = 3
n_forwards = 24 * days_forward
n_backwards = 24 * days_backward
th1_low = 0.
th2_low = 5.
th1_mid = 5.
th2_mid = 7.
th1_high = 7.
th2_high = 9.
th1_all = 0.
th2_all = 9.


x1_train = (dataset_dict_all['state_data'])[:, :-shift]
x2_train = (dataset_dict_all['state_data'])[:, shift:]
u_train = (dataset_dict_all['drivers_data'])[:, :-shift]

print(f'---------x_train_svd_obj:{ get_size(x_train_svd_obj, seen=None)}--------------------------------')

x1_train_centered = u.normalize_with_dict(x1_train, x_train_dict, centering_method)
x2_train_centered = u.normalize_with_dict(x2_train, x_train_dict, centering_method)


print(f'\n Dataset with {segment} magnetic storms \n')
dataset_dict = {}
if segment == 'quiet':
    th1 = th1_low
    th2 = th2_low
    idx_quiet_train = u.select_indices(dataset_dict_all['drivers_train'][kp_idx,:].reshape((1, -1)), th1_low, th2_low, 0, 0)[:-1]
    idx_train_val = np.copy(idx_quiet_train)
    idx_train_val_subsets = np.array_split(idx_train_val, (idx_train_val.shape[0] + n_days * 24 - 1) // (n_days * 24), axis = 0) 

    idx_quiet_test = u.select_indices(dataset_dict_all['drivers_test'][kp_idx,:].reshape((1, -1)), th1_low, th2_low, n_backwards, n_forwards)[:-1]
    idx_test = np.copy(idx_quiet_test)
    idx_test_subsets = np.array_split(idx_test, (idx_test.shape[0] + n_days * 24 - 1) // (n_days * 24), axis = 0) 

elif segment == 'moderate':
    th1 = th1_mid
    th2 = th2_mid
    idx_moderate_train = u.select_indices(dataset_dict_all['drivers_train'][kp_idx,:].reshape((1, -1)), th1_mid, th2_mid, n_backwards, n_forwards)
    idx_train_val = np.copy(idx_moderate_train)
    idx_train_val_subsets = np.split(idx_train_val, np.where(np.diff(idx_train_val) > 1)[0] + 1)

    idx_moderate_test = u.select_indices(dataset_dict_all['drivers_test'][kp_idx,:].reshape((1, -1)), th1_mid, th2_mid, n_backwards, n_forwards)
    idx_test = np.copy(idx_moderate_test)
    idx_test_subsets = np.split(idx_test, np.where(np.diff(idx_test) > 1)[0] + 1)

    
elif segment == 'strong':
    th1 = th1_high
    th2 = th2_high
    idx_strong_train = u.select_indices(dataset_dict_all['drivers_train'][kp_idx,:].reshape((1, -1)), th1_high, th2_high, n_backwards, n_forwards)
    idx_train_val = np.copy(idx_strong_train)
    idx_train_val_subsets = np.split(idx_train_val, np.where(np.diff(idx_train_val) > 1)[0] + 1)

    idx_strong_test = u.select_indices(dataset_dict_all['drivers_test'][kp_idx,:].reshape((1, -1)), th1_high, th2_high, n_backwards, n_forwards)
    idx_test = np.copy(idx_strong_test)
    idx_test_subsets = np.split(idx_test, np.where(np.diff(idx_test) > 1)[0] + 1)

elif segment == 'all':
    th1 = th1_all
    th2 = th2_all
    idx_all_train = u.select_indices(dataset_dict_all['drivers_data'][kp_idx,:].reshape((1, -1)), th1_all, th2_all, 0, 0)[:-1]
    idx_train_val = np.copy(idx_all_train)
    idx_train_val_subsets = np.array_split(idx_train_val, (idx_train_val.shape[0] + n_days * 24 - 1) // (n_days * 24), axis = 0) 
    idx_all_test = u.select_indices(dataset_dict_all['drivers_data'][kp_idx,:].reshape((1, -1)), th1_all, th2_all, 0, 0)[:-1]
    idx_test = np.copy(idx_all_test)
    idx_test_subsets = np.array_split(idx_test, (idx_test.shape[0] + n_days * 24 - 1) // (n_days * 24), axis = 0) 
    

# %%

# %%
mdl_stats_lst = []
gamma = 1./1.
sub_intervals = 60 #integer number saying how many time steps are required to make 1 time step of the original discrete system
delta_t = gamma*sub_intervals
pca_coupling = [2]

z1_train = np.copy(x_train_svd_obj.U.T @ x1_train_centered)    
input_features = ['x_'+ str(k+1).zfill(2) for k in range(z1_train.shape[0])]

###################################### N days dynamic forecats on train ######################################

state_subsets = []
for idx_array in idx_train_val_subsets:
    state_subsets.append(x1_train_centered[:, idx_array])

drivers_subsets = []
for idx_array in idx_train_val_subsets:
    drivers_subsets.append(u_train[:, idx_array])


ys_df_results = []
ys_continuous_df_results = []
mdl_stats_dict = {}
df_mae_lst = []
for alpha_ridge in ridge_parameters: 
    print(f'Ridge parameter: {alpha_ridge} ')
    for model_name in models_dict.keys():
        if (model_name == 'nl-dmd') & (alpha_ridge != ridge_parameters[0]):
            continue
        model_params['model'] = model_name
        print(model_name)
        for basis_name in models_dict[model_name].keys():
            print(basis_name)
            if model_name == 'dmd':
                Y_reg_norm_dict = Y_reg_norm_dict_dmd
                normalization_method = 'std'
            elif model_name == 'nl-dmd':
                Y_reg_norm_dict = Y_reg_norm_dict_nl_dmd
                normalization_method = 'std'                
            elif model_name == 'sindy':
                normalization_method = 'std'
                Y_reg_norm_dict_sindy = bf_normalization_dict[basis_name]['Y_reg_norm_dict_sindy']
                X_reg_norm_dict_sindy = bf_normalization_dict[basis_name]['X_reg_norm_dict_sindy']
                Y_reg_norm_dict = Y_reg_norm_dict_sindy
    
            
            if model_name == 'dmd':
                B_dmd = models_dict[model_name][basis_name][f'ridge_parameter_{alpha_ridge:.2f}']
                B = np.copy(B_dmd)
            elif model_name == 'sindy':
                B_sindy_sm_discrete = models_dict[model_name][basis_name][f'ridge_parameter_{alpha_ridge:.2f}']['sm']
                B_sindy_sm_kp_discrete = models_dict[model_name][basis_name][f'ridge_parameter_{alpha_ridge:.2f}']['sm_kp']
                B_sindy_f10_discrete = models_dict[model_name][basis_name][f'ridge_parameter_{alpha_ridge:.2f}']['sm_f10']
                B_sindy_joint_discrete = models_dict[model_name][basis_name][f'ridge_parameter_{alpha_ridge:.2f}']['joint']
                B_sindy_combined_discrete = models_dict[model_name][basis_name][f'ridge_parameter_{alpha_ridge:.2f}']['combined']
                model_params['functions'] = basis_functions_dict[basis_name]
                sindy_tgt_col = B_sindy_combined_discrete.shape[1] - n_components + pca_coupling[0]
                
                array_joint = move_column(np.copy(B_sindy_joint_discrete), 5, sindy_tgt_col)
                A_sindy_joint = np.copy(array_joint[:, -10:])
                B_sindy_joint = np.copy(array_joint[:, :(-10)])

                phi_sindy_joint = logm(np.block([
                                        [A_sindy_joint, B_sindy_joint],
                                        [np.zeros((B_sindy_joint.shape[1], B_sindy_joint.shape[0])), np.eye((B_sindy_joint.shape[1]))]
                                    ]))/delta_t
                A_sindy_joint_c = phi_sindy_joint[:A_sindy_joint.shape[0], :A_sindy_joint.shape[1]]
                B_sindy_joint_c = phi_sindy_joint[:B_sindy_joint.shape[0], A_sindy_joint.shape[1]:]

                array_f10 = move_column(np.copy(B_sindy_f10_discrete), 5, sindy_tgt_col)
                A_sindy_f10 = np.copy(array_f10[:, -10:])
                B_sindy_f10 = np.copy(array_f10[:, :(-10)])

                phi_sindy_f10 = logm(np.block([
                                        [A_sindy_f10, B_sindy_f10],
                                        [np.zeros((B_sindy_f10.shape[1], B_sindy_f10.shape[0])), np.eye((B_sindy_f10.shape[1]))]
                                    ]))/delta_t
                A_sindy_f10_c = phi_sindy_f10[:A_sindy_f10.shape[0], :A_sindy_f10.shape[1]]
                B_sindy_f10_c = phi_sindy_f10[:B_sindy_f10.shape[0], A_sindy_f10.shape[1]:]                

                del array_joint, A_sindy_joint, B_sindy_joint, array_f10, A_sindy_f10, B_sindy_f10
            elif model_name == 'nl-dmd':
                B_nl_dmd = models_dict[model_name][basis_name][f'ridge_parameter_{alpha_ridge:.2f}']
                B = np.copy(B_nl_dmd)
                array = np.copy(B_nl_dmd)
                A_dmd = np.copy(np.copy(B_nl_dmd[:, 4:(-4)]))
                B_dmd = np.copy(np.concatenate([B_nl_dmd[:, :4], B_nl_dmd[:, (-4):]], axis = 1))
                delta_t = gamma*sub_intervals
                # A_dmd_c, B_dmd_c = discrete_to_continuous(A_dmd, B_dmd, delta_t)

                phi_dmd = logm(np.block([
                                        [A_dmd, B_dmd],
                                        [np.zeros((B_dmd.shape[1], B_dmd.shape[0])), np.eye((B_dmd.shape[1]))]
                                    ]))/delta_t
                A_dmd_c = phi_dmd[:A_dmd.shape[0], :A_dmd.shape[1]]
                B_dmd_c = phi_dmd[:B_dmd.shape[0], A_dmd.shape[1]:]
                del array, A_dmd, B_dmd
            
            yk_lst = []
            windows_lst = []
            continuous_windows_lst = []
            ys_lst = []
            frcst_window_length_lst = []
            cont_time_lst = []
            for n, state_n in enumerate(state_subsets):
                # n = 19
                # state_n = state_subsets[n]
                state_n_size = state_n.shape[1]
                z1_k = np.copy(x_train_svd_obj.U.T @ state_n[:, 0].reshape((-1, 1)))
                copied_drivers = np.copy(drivers_subsets[n])
                interpolated_drivers = interpolate_matrix_rows(copied_drivers, sub_intervals)
                delta_f10_0 = 0
                
                if model_params['model'] == 'sindy':
                    k = 0 
                    # t_span = (0, sub_intervals*tuple(range(drivers_subsets[n].shape[1]))[-1])
                    # t_interval = np.linspace(t_span[0], t_span[1], (sub_intervals)*((t_span[1] - t_span[0]) + 1))
                    t_span = (0, sub_intervals*(tuple(range(copied_drivers.shape[1]))[-1] + 1) - 1)
                    t_interval = np.linspace(t_span[0], t_span[1], ((t_span[1] - t_span[0]) + 1))

                    X_k_norm = build_sindy_dyn_frcst_inputs(z1_k, interpolated_drivers, X_reg_norm_dict_sindy, pca_coupling, kp_idx, f10_idx, model_params,\
                        normalization_method, input_features, k = int(k))   
                    qF_norm = move_column(np.copy(X_k_norm).T, 5, sindy_tgt_col).T
                    q0_norm = np.copy(qF_norm[-10:])
                    
                    solution = solve_ivp(
                        ode_func_sindy,
                        t_span,
                        q0_norm.flatten(),
                        args = (interpolated_drivers, A_sindy_joint_c, B_sindy_joint_c, A_sindy_f10_c, B_sindy_f10_c, sindy_tgt_col),
                        method = 'RK45',
                        t_eval = t_interval
                    )
            
                    t = solution.t
                    
                    q_sol = np.full((len(q0_norm.flatten()), len(t_interval)), np.nan)
                    q_sol[:, :len(solution.t)] = solution.y
                    q_sol_denormalized = q_sol[:, ::sub_intervals] * Y_reg_norm_dict_sindy['x_std'] + Y_reg_norm_dict_sindy['x_mean']
                    del solution, interpolated_drivers, q0_norm, qF_norm, t_interval, z1_k

                    # print(f'State {n}')
                    # print(f'delta shots{q_sol_denormalized.shape[1] - drivers_subsets[n].shape[1]}')
                        
                elif model_params['model'] == 'nl-dmd':  
                    k = 0
                    # t_span = (0, sub_intervals*tuple(range(drivers_subsets[n].shape[1]))[-1])
                    # t_interval = np.linspace(t_span[0], t_span[1], (sub_intervals)*((t_span[1] - t_span[0]) + 1))                    
                    t_span = (0, sub_intervals*(tuple(range(copied_drivers.shape[1]))[-1] + 1) - 1)
                    t_interval = np.linspace(t_span[0], t_span[1], ((t_span[1] - t_span[0]) + 1))

                    X_k = np.concatenate([z1_k, interpolated_drivers[f10_idx:, k].reshape((-1, 1)), (interpolated_drivers[f10_idx, k] * interpolated_drivers[kp_idx, k]).reshape((-1, 1)), 
                        (interpolated_drivers[kp_idx, k] * interpolated_drivers[kp_idx, k]).reshape((-1, 1))])
                    X_k_norm = u.normalize_with_dict(X_k, X_reg_norm_dict_nl_dmd, normalization_method)   
                    X_k_norm = np.concatenate([interpolated_drivers[1:f10_idx, k].reshape((-1, 1)), X_k_norm])
                    q0_norm = np.copy(X_k_norm[4:(-4), :])

                    solution = solve_ivp(
                        ode_func_dmd,
                        t_span,
                        q0_norm.flatten(),
                        args = (interpolated_drivers, A_dmd_c, B_dmd_c),
                        method = 'RK45',
                        t_eval = t_interval
                    )
                    
                    t = solution.t
                    q_sol = np.full((len(q0_norm.flatten()), len(t_interval)), np.nan)
                    q_sol[:, :len(solution.t)] = solution.y
                    q_sol_denormalized = q_sol[:, ::sub_intervals] * Y_reg_norm_dict_nl_dmd['x_std'] + Y_reg_norm_dict_nl_dmd['x_mean']
                    del solution, interpolated_drivers, q0_norm, t_interval, z1_k

                cont_time_lst.append(q_sol_denormalized)

                
                z1_k = np.copy(x_train_svd_obj.U.T @ state_n[:, 0].reshape((-1, 1)))
                q0_discrete = np.copy(z1_k)
                frcst_lst = []
                for k in range(copied_drivers.shape[1]):  
                    
                    if model_params['model'] == 'sindy':
                        X_k_for_sindy = np.concatenate([z1_k[pca_coupling].reshape((1, 1)), copied_drivers[f10_idx:, k].reshape((-1, 1))])
                        current_kp = copied_drivers[kp_idx, k]
                        library_dict = u.create_library_functions(np.copy(X_k_for_sindy.T), model_params['functions'], input_features)
                        theta_k = library_dict['theta'].T
                        del library_dict
                        X_k = np.concatenate([theta_k, np.delete(z1_k, pca_coupling, axis = 0)], axis = 0)     
                        X_k_norm = u.normalize_with_dict(X_k[1:], X_reg_norm_dict_sindy, method = normalization_method)  
                        X_k_norm = np.concatenate([X_k[0, :].reshape((1, -1)), copied_drivers[1:f10_idx, k].reshape((-1, 1)), X_k_norm])
                        # Y_k = u.dynamic_prediction_dual_regimes(X_k_norm, current_kp, delta_f10, B_nl00, B_nl10, B_nl20, B_nl30, B_nl40, B_nl50, B_nl60, B_nl70, B_nl80,\
                            # B_nl01, B_nl11, B_nl21, B_nl31, B_nl41, B_nl51, B_nl61, B_nl71, B_nl81)
                        #Y_k = B_sindy_sm_discrete @ X_k_norm
                        # Y_k = B_sindy_sm_kp_discrete @ X_k_norm
                        # Y_k = (B_sindy_joint_discrete * (current_kp >= 3) + B_sindy_f10_discrete * (current_kp < 3)) @ X_k_norm
                        # Y_k = B_sindy_combined_discrete @ X_k_norm
                        Y_k = (B_sindy_joint_discrete * (current_kp >= 3) + B_sindy_f10_discrete * (current_kp < 3)) @ X_k_norm
                    elif model_params['model'] == 'dmd': 
                        X_k = np.concatenate([z1_k, copied_drivers[f10_idx:, k].reshape((-1, 1))])
                        X_k_norm = u.normalize_with_dict(X_k, X_reg_norm_dict_dmd, normalization_method)
                        X_k_norm = np.concatenate([copied_drivers[1:f10_idx, k].reshape((-1, 1)), X_k_norm])
                        Y_k = B @ X_k_norm 
                    elif model_params['model'] == 'nl-dmd': 
                        X_k = np.concatenate([z1_k, copied_drivers[f10_idx:, k].reshape((-1, 1)), (copied_drivers[f10_idx, k] * copied_drivers[kp_idx, k]).reshape((-1, 1)), 
                            (copied_drivers[kp_idx, k] * copied_drivers[kp_idx, k]).reshape((-1, 1))])
                        X_k_norm = u.normalize_with_dict(X_k, X_reg_norm_dict_nl_dmd, normalization_method)
                        X_k_norm = np.concatenate([copied_drivers[1:f10_idx, k].reshape((-1, 1)), X_k_norm])
                        Y_k = B @ X_k_norm 
                    
                    # Y_k_denormalized = u.denormalize(Y_k, Y_reg_norm_dict, normalization_method)
        
                    Y_k_denormalized = Y_k * Y_reg_norm_dict['x_std'] + Y_reg_norm_dict['x_mean'] 
                    
                    yk_lst.append(np.copy(Y_k_denormalized))
                    frcst_lst.append(np.copy(Y_k_denormalized)) 
                    z1_k = np.copy(np.copy(Y_k_denormalized))
                    del Y_k_denormalized
                
                del copied_drivers
                frcst_window_length_lst.append(np.arange(state_n.shape[1]).reshape((-1, 1)))
                window_n = np.float64(np.concatenate(frcst_lst, axis = 1))
                windows_lst.append(window_n) 

            if model_params['model'] == 'sindy':
                del A_sindy_joint_c, B_sindy_joint_c, A_sindy_f10_c, B_sindy_f10_c
            elif  model_params['model'] == 'nl-dmd':
                del A_dmd_c, B_dmd_c
            frcst_periods = np.concatenate(frcst_window_length_lst, axis = 0)
            ys = np.concatenate(yk_lst, axis = 1)
            ys_lst.append(ys)
            full_frcst = np.copy(np.concatenate(windows_lst, axis = 1))
            continuous_full_frcst = np.copy(np.concatenate(cont_time_lst, axis = 1))
            predicted_state = x_train_svd_obj.U @ full_frcst + x_train_dict['x_mean']
            continuous_predicted_state = x_train_svd_obj.U @ continuous_full_frcst + x_train_dict['x_mean']
        
            rho_fcst = np.float64(10**predicted_state)
            rho_fcst_continuous = np.float64(10**continuous_predicted_state)
            rho_actual = np.float64(10**x2_train[:, idx_train_val])
            error = rho_fcst - rho_actual
            error_continuous = rho_fcst_continuous - rho_actual
            pae = np.abs(error)/rho_actual
            pae_continuous = np.abs(error_continuous)/rho_actual
            instant_pae = np.mean(pae, axis = 0, keepdims = True)
            instant_pae_continuous = np.mean(pae_continuous, axis = 0, keepdims = True)     
            
    
            # df_ys = pd.DataFrame(np.concatenate([frcst_periods, \
            #     u_train[0, idx_train_val].T.reshape((-1, 1)), doy_all_train[idx_train_val].reshape((-1, 1)), \
            #         u_train[kp_idx, idx_train_val].T.reshape((-1, 1)), u_train[f10_idx, idx_train_val].T.reshape((-1, 1)), \
            #             np.sum(error, axis = 0, keepdims = True).T, np.sum(rho_actual, axis = 0, keepdims = True).T, instant_pae.T.reshape((-1, 1)), ys.T], axis = 1 ), \
            #                 columns = ['period', 'year', 'doy', 'kp', \
            #                     'f107', 'error', 'actual', 'mape'] + [f'y_{k}' for k in range(ys.shape[0])])
            # df_ys['ones'] = 1
            # df_ys['model'] = model_name
            # df_ys['alpha_ridge'] = alpha_ridge
            # df_ys['basis_functions'] = basis_name
            # df_ys['cycle'] = (df_ys['period'] == 0).cumsum() - 1
            # df_ys['mdl'] = model_name
            # df_ys['segment'] = segment
            # ys_df_results.append(df_ys)

            # df_ys_continuous = pd.DataFrame(np.concatenate([frcst_periods, \
            #     u_train[0, idx_train_val].T.reshape((-1, 1)), doy_all_train[idx_train_val].reshape((-1, 1)), \
            #         u_train[kp_idx, idx_train_val].T.reshape((-1, 1)), u_train[f10_idx, idx_train_val].T.reshape((-1, 1)), \
            #             np.sum(error, axis = 0, keepdims = True).T, np.sum(rho_actual, axis = 0, keepdims = True).T, instant_pae.T.reshape((-1, 1)), \
            #                 continuous_full_frcst.T], axis = 1 ), \
            #                     columns = ['period', 'year', 'doy', 'kp', \
            #                         'f107', 'error', 'actual', 'mape'] + [f'y_{k}' for k in range(ys.shape[0])])
            # df_ys_continuous['ones'] = 1
            # df_ys_continuous['model'] = model_name
            # df_ys_continuous['alpha_ridge'] = alpha_ridge
            # df_ys_continuous['basis_functions'] = basis_name
            # df_ys_continuous['cycle'] = (df_ys['period'] == 0).cumsum() - 1
            # df_ys_continuous['mdl'] = model_name
            # df_ys_continuous['segment'] = segment
            # ys_continuous_df_results.append(df_ys)
        
        
            df_mae = pd.DataFrame(np.concatenate([frcst_periods, \
                u_train[0, idx_train_val].T.reshape((-1, 1)), doy_all_train[idx_train_val].reshape((-1, 1)), \
                    u_train[kp_idx, idx_train_val].T.reshape((-1, 1)), u_train[f10_idx, idx_train_val].T.reshape((-1, 1)), \
                        np.sum(error, axis = 0, keepdims = True).T, np.sum(rho_actual, axis = 0, keepdims = True).T, instant_pae.T.reshape((-1, 1)), \
                             instant_pae_continuous.T.reshape((-1, 1))], axis = 1 ), \
                            columns = ['period', 'year', 'doy', 'kp', \
                                'f107', 'error', 'actual', 'mape', 'mape_continuous'])
            del instant_pae_continuous, pae_continuous, pae, error_continuous, error, rho_actual, rho_fcst_continuous, \
                rho_fcst, continuous_predicted_state, predicted_state, continuous_full_frcst, full_frcst
            
            df_mae['ones'] = 1
            overall_stats = df_mae.loc[(df_mae.kp >= th1) & (df_mae.kp < th2)][['kp', 'f107', 'mape', 'mape_continuous']].mean()
            print(f'{n_days} days dynamic forecast on test dataset for {model_name}, with {instant_pae.shape} samples')
            print(f'{model_name} overall MAPE:{100. * overall_stats['mape']:.2f}%, overall MAPE cont:{100. * overall_stats['mape_continuous']:.2f}%')
            mape_year = df_mae.loc[(df_mae.kp >= th1) & (df_mae.kp < th2)].groupby(['year']).agg({'period':"mean" , 'kp':"mean" , \
                                'f107':"mean" , 'error':"mean" , 'actual':"mean" , 'mape':"mean", 'mape_continuous':"mean" , 'ones':"sum" }).reset_index()
            mape_year.mape = 100. * mape_year.mape
            mape_year.mape_continuous = 100. * mape_year.mape_continuous
            print(f'{model_name} MAPE (%) per year:\n')
            print(mape_year.drop(columns = 'period').round(2))
            mape = overall_stats['mape']
            mdl_stats_dict.update({f'{model_name}': {'mape': mape, 'instant_pae': instant_pae}})
            mdl_stats_lst.append(mdl_stats_dict)
            df_mae['alpha_ridge'] = alpha_ridge
            df_mae['basis_functions'] = basis_name
            df_mae['cycle'] = (df_mae['period'] == 0).cumsum() - 1
            df_mae['mdl'] = model_name
            df_mae['segment'] = segment
            df_mae_lst.append(df_mae)
            u.save_object(df_mae, f'outputs/stats/{segment}/mae_train_val_{n_days}d_frcst_{model_name}_{segment}.pkl')
            df_to_plot = df_mae.groupby(['period']).agg({'mape':'mean'}).reset_index()
            del instant_pae
            
df_mae_all_train = pd.concat(df_mae_lst).reset_index()
df_mae_all_train.to_csv(f'outputs/stats/{segment}/df_mae_all_train.csv', index = False)
# ys_df_train = pd.concat(ys_df_results).reset_index()
# u.save_object(ys_df_train, f'outputs/stats/{segment}/ys_train_val_{n_days}d_frcst_{model_name}_{segment}.pkl')

if len(models) >= 2:
    mape_percent_change = 100 * (mdl_stats_dict['sindy']['mape'] - mdl_stats_dict['nl-dmd']['mape']) / mdl_stats_dict['nl-dmd']['mape']
    print(f'Relative percent improvement between SINDY-c and DMD-c MAPE on train-val is {mape_percent_change:.2f}')            

# agg_values = ys_df_train.groupby(['period']).agg({'ones':'sum', 'kp':'mean', 'mape':['median', 'std']}).reset_index().droplevel(axis=1, level=[0]).fillna(0)


# %%

# %%

# %%

# %%

# %%
-trip transportation between
Embassy Suites and Mesa Lab will be provided on a 

# %%
