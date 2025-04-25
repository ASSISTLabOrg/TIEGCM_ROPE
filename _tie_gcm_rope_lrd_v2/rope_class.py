import numpy as np
import orekit
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
import utilities_ds as u




class SvdContainer:
    def __init__(self, u_svd, mu):
        self.U = u_svd
        self.mu = mu

class rope_data:
    '''
    Class to store and propagate tyhe SINDY model based on TIEGCM physics based data
    '''
    def __init__( self, datapath: str = ".", drivers = None):

        self.input_data_sindy = np.load(path.join(datapath, 'z_drivers_dataset_lrd_96_01.npz'), allow_pickle=True)
        self.U0 = self.input_data_sindy['u_svd'].reshape((24, 20, 16, 10), order='C')
        self.mu0 = self.input_data_sindy['mu_svd'].reshape((24, 20, 16), order='C')
        self.models_coefficients = self.input_data_sindy['models_coefficients'][()]
        self.initial_conditions = pd.DataFrame(self.input_data_sindy['initial_conditions'][()], columns = [ 'f10', 'kp']+[f'z_{str(k).zfill(2)}' for k in range(10)])
        if drivers is None:
            self.drivers = self.input_data_sindy['drivers']
        else:
            self.drivers = drivers
    
        self.X_reg_norm_dict_nl_dmd = self.input_data_sindy['models_coefficients'][()]['X_reg_norm_dict_nl_dmd']
        self.Y_reg_norm_dict_nl_dmd = self.input_data_sindy['models_coefficients'][()]['Y_reg_norm_dict_nl_dmd']
        self.x_train_svd_obj = SvdContainer(self.input_data_sindy['u_svd'], self.input_data_sindy['mu_svd'])
        self.x_train_svd_obj.norm_dict = {'x_mean': self.input_data_sindy['mu_svd']}
        self.normalization_method = 'std'
        T = [15.34175, 12.2734, 24., 48., 20.45566667, 30.6835, 61.367]
        self.delta_rho_ic = 6
        self.f10_idx = 5
        self.kp_idx = 6
        self.pca_coupling = 2
        self.input_features = ['x_'+ str(k+1).zfill(2) for k in range(10)]
        poly1 = {'p1': lambda x: x}
        poly2 = {'p2': lambda x: x**2}
        poly3 = {'p3': lambda x: x**3}
        poly5 = {'p3': lambda x: x**5}
        poly7 = {'p3': lambda x: x**7}
        exp1 = {'e1': lambda x: np.exp(-x)}
        sincos4 = {'g13': lambda x: np.sin(2*np.pi*x/T[3]), 'g14': lambda x: np.cos(2*np.pi*x/T[3])}
        sincos7 = {'g13': lambda x: np.sin(2*np.pi*x/T[6]), 'g14': lambda x: np.cos(2*np.pi*x/T[6])}

        self.basis_functions_dict = {
        'poly': poly1,
        'poly13': poly1 | poly3,
        'poly135': poly1 | poly3 | poly5, 
        'poly7': poly7,
        'poly1357': poly1 | poly3 | poly5 | poly7
        }
        self.selected_bf_dict = {
            'poly': 0.,
            'poly': 10.,
            'poly13': 0.0,
            'poly13': 0.5,
            'poly135': 10., 
            'poly1357': 0.5,
            'poly7': 0.0,
        }


    def move_column(self, array, from_col, to_col):
        return np.insert(np.delete(array, from_col, axis=1), to_col, array[:, from_col], axis=1)
        
    def build_sindy_dyn_frcst_inputs(self, z1_k, drivers, X_reg_norm_dict_sindy, pca_coupling, kp_idx, f10_idx, model_params, normalization_method, input_features, k = 0):
        X_k_for_sindy = np.concatenate([z1_k[pca_coupling].reshape((1, 1)), drivers[f10_idx:, k].reshape((-1, 1))])
        library_dict = u.create_library_functions(np.copy(X_k_for_sindy.T), model_params['functions'], input_features)
        theta_k = library_dict['theta'].T
        X_k = np.concatenate([theta_k, np.delete(z1_k, pca_coupling, axis = 0)], axis = 0)     
        X_k_norm = u.normalize_with_dict(X_k[1:], X_reg_norm_dict_sindy, method = normalization_method)  
        X_k_norm = np.concatenate([X_k[0, :].reshape((1, -1)), drivers[1:f10_idx, k].reshape((-1, 1)), X_k_norm]) 

        return X_k_norm

    def ode_func_sindy(self, t, q_norm, drivers, A_sindy_joint_c, B_sindy_joint_c, A_sindy_f10_c, B_sindy_f10_c, sindy_tgt_col):
        discrete_idx = np.searchsorted(self.t_interval, t)
        discrete_idx = max(0, min(int(discrete_idx), len(self.t_interval) - 1)) 
        
        current_kp = drivers[self.kp_idx, int(t)]
        
        q_denormalized = u.denormalize(q_norm.reshape((-1, 1)), self.Y_reg_norm_dict_sindy, self.normalization_method)
        X_k_norm = self.build_sindy_dyn_frcst_inputs(q_denormalized, drivers, self.X_reg_norm_dict_sindy, self.pca_coupling, self.kp_idx, self.f10_idx, self.model_params,\
            self.normalization_method, self.input_features, k = int(t)) 
        qF_norm = self.move_column(np.copy(X_k_norm).T, 5, sindy_tgt_col).T    
        F_norm = np.copy(qF_norm[:(-10), 0]).reshape((-1, 1))
        dq_dt = ((A_sindy_joint_c * (current_kp >= 3) + A_sindy_f10_c * (current_kp < 3)) @ q_norm.reshape((-1, 1)) + (B_sindy_joint_c * (current_kp >= 3) + B_sindy_f10_c * (current_kp < 3)) @ F_norm.reshape((-1, 1))).flatten()   
        return dq_dt


    def ode_func_dmd(self, t, q_norm, drivers, A_c, B_c):

        discrete_idx = np.searchsorted(self.t_interval, t)
        discrete_idx = max(0, min(int(discrete_idx), len(self.t_interval) - 1)) 
        k = int(t)
        
        q_denormalized = u.denormalize(q_norm.reshape((-1, 1)), self.Y_reg_norm_dict_nl_dmd, self.normalization_method)
        
        X_k = np.concatenate([q_denormalized, drivers[self.f10_idx:, k].reshape((-1, 1)), (drivers[self.f10_idx, k] * drivers[self.kp_idx, k]).reshape((-1, 1)), 
            (drivers[self.kp_idx, k] * drivers[self.kp_idx, k]).reshape((-1, 1))])
        X_k_norm = u.normalize_with_dict(X_k, self.X_reg_norm_dict_nl_dmd, self.normalization_method)
        X_k_norm = np.concatenate([drivers[1:self.f10_idx, k].reshape((-1, 1)), X_k_norm])
        q0_norm = np.copy(X_k_norm[4:(-4), :])
        F_norm = np.copy(np.concatenate([X_k_norm[:4, :], X_k_norm[(-4):, :]], axis = 0))
        dq_dt = (A_c @ q_norm.reshape((-1, 1)) + B_c @ F_norm.reshape((-1, 1))).flatten()   
        return dq_dt

    def interpolate_matrix_rows(self, matrix, sub_intervals):
        n, m = matrix.shape
        interpolated_columns = m * sub_intervals 
        result = np.zeros((n, interpolated_columns))
        
        for i in range(n):
            x_original = np.arange(m)
            x_interpolated = np.linspace(0, m - 1, interpolated_columns)
            
            result[i, :] = np.interp(x_interpolated, x_original, matrix[i, :])
        
        return result
    

    def find_closest_match(self, df, f10_target, kp_target):
        # Compute absolute differences
        df['f10_diff'] = np.abs(df['f10'] - f10_target)
        df['kp_diff'] = np.abs(df['kp'] - kp_target)
        
        # Find the row with the smallest sum of differences
        closest_row = df.loc[(df['f10_diff'] + df['kp_diff']).idxmin()]
        closest_row = closest_row.drop(columns=['f10_diff', 'kp_diff'])
        # Drop helper columns before returning
        df = df.drop(columns=['f10_diff', 'kp_diff'])
        
        return closest_row



    def get_initial_z_from_drivers(self, sampled_ic_table, f10_input, kp_input):

        f10_sorted = np.sort(sampled_ic_table['f10'].unique())
        f10_below = f10_sorted[f10_sorted <= f10_input].max() #if any(f10_sorted <= f10_input) #else f10_sorted[f10_sorted <= f10_input].max()
        kp_sorted = np.sort(sampled_ic_table['kp'].unique())
        kp_below = kp_sorted[kp_sorted <= kp_input].max() #if any(kp_sorted <= kp_input) else kp_sorted[kp_sorted <= kp_input].max()
        if f10_below is None or kp_below is None:
            return None
        
        row = self.find_closest_match(sampled_ic_table, f10_below, kp_below)
        
        return row.iloc[2:12].squeeze() if not row.empty else None

    def propagate_models(self, init_date, forward_propagation = 5):

        '''
        Propagation of the density models
        
        Inputs:
          init_date: datetime64[ns], date representing the initial propagation time
          forward_propagation: int representing the number of days the density is propagated since the chosen year-day_of_year
        
        Outputs:
          No outputs. Saves results within the object itself.
        '''
        init_date = pd.to_datetime(init_date)

        self.sub_intervals = 60
        year = init_date.year
        day_of_year = init_date.day_of_year#use date 

        start_date = init_date
        end_date = datetime(year, 1, 1) + timedelta(days=day_of_year + forward_propagation - 1)

        # Generate date series with 10-second intervals for 15 days
        self.date_series = pd.date_range(start=start_date, end = end_date, freq = str(self.sub_intervals) + 's')[:(-1)]

        n_components = 10
        
        normalization_method = 'std'
        gamma = 1
        delta_t = gamma*self.sub_intervals 
        t0 = day_of_year * 24 - 24 + self.delta_rho_ic # 24 hours of day doy have passed, so you'd start from doy + 1 without subtracting 24
        n_days_frcst = forward_propagation
        forward_hours = n_days_frcst * 24
        tf = t0 + forward_hours + self.delta_rho_ic #remove this since you observed sindy is early by 6 hours
        f10_idx = 5
        kp_idx = 6
        input_data_models = self.input_data_sindy['models_coefficients'][()]
        B_nl_dmd_discrete = input_data_models['models_dict']['nl-dmd']['plain']['ridge_parameter_0.50']

        print(f'Maximum available time T = {(self.sub_intervals*24*n_days_frcst-1)*60}s')
        self.z_results_lst = []
        for chosen_basis_function in list(self.selected_bf_dict.keys()):
            ridge_label = 'ridge_parameter_' + "{:.2f}".format(self.selected_bf_dict[chosen_basis_function])

            self.B_sindy_combined_discrete = input_data_models['models_dict']['sindy'][chosen_basis_function][ridge_label]['combined']
            self.B_sindy_joint_discrete = input_data_models['models_dict']['sindy'][chosen_basis_function][ridge_label]['joint']
            self.B_sindy_f10_discrete = input_data_models['models_dict']['sindy'][chosen_basis_function][ridge_label]['sm_f10']
            self.X_reg_norm_dict_sindy = input_data_models['bf_normalization_dict'][chosen_basis_function]['X_reg_norm_dict_sindy']
            self.Y_reg_norm_dict_sindy = input_data_models['bf_normalization_dict'][chosen_basis_function]['Y_reg_norm_dict_sindy']
            self.model_params = {'normalization_method': 'std', 'functions': self.basis_functions_dict[chosen_basis_function]}
            self.sindy_tgt_col = self.B_sindy_combined_discrete.shape[1] - n_components + self.pca_coupling

            IC_idx = np.where(self.drivers[0,:] == year)[0]
            
            biased_ic_indices = np.arange(np.min(IC_idx) - self.delta_rho_ic, np.min(IC_idx) + tf)
            drivers_IC = np.copy(self.drivers[:, biased_ic_indices])
            
            
            drivers = np.copy(drivers_IC[:, (t0):(tf)])
            
            interpolated_drivers = self.interpolate_matrix_rows(drivers, self.sub_intervals)
            if not hasattr(self, 'propagation_drivers'):
                self.t0 = t0
                self.tf = tf
                self.propagation_drivers = interpolated_drivers

            f10_value = np.copy(drivers_IC[5, t0])
            kp_value = np.copy(drivers_IC[6, t0])
            
            z_series = self.get_initial_z_from_drivers(self.initial_conditions, f10_value, kp_value)
            
            z1_k = z_series.values.reshape((10, 1))
            
            input_features = ['x_'+ str(k+1).zfill(2) for k in range(z1_k.shape[0])]
            
            k = 0
            
            t_span = (0, self.sub_intervals*(tuple(range(drivers.shape[1]))[-1] + 1) - 1)
            t_interval = np.linspace(t_span[0], t_span[1], ((t_span[1] - t_span[0]) + 1))
            self.t_interval = t_interval[self.sub_intervals*self.delta_rho_ic:]
            
            
            ###########################################sindy###############################################

            array_joint = self.move_column(np.copy(self.B_sindy_joint_discrete), 5, self.sindy_tgt_col)
            A_sindy_joint = np.copy(array_joint[:, -10:])
            B_sindy_joint = np.copy(array_joint[:, :(-10)])

            phi_sindy_joint = logm(np.block([
                                    [A_sindy_joint, B_sindy_joint],
                                    [np.zeros((B_sindy_joint.shape[1], B_sindy_joint.shape[0])), np.eye((B_sindy_joint.shape[1]))]
                                ]))/delta_t
            A_sindy_joint_c = phi_sindy_joint[:A_sindy_joint.shape[0], :A_sindy_joint.shape[1]]
            B_sindy_joint_c = phi_sindy_joint[:B_sindy_joint.shape[0], A_sindy_joint.shape[1]:]

            array_f10 = self.move_column(np.copy(self.B_sindy_f10_discrete), 5, self.sindy_tgt_col)
            A_sindy_f10 = np.copy(array_f10[:, -10:])
            B_sindy_f10 = np.copy(array_f10[:, :(-10)])

            phi_sindy_f10 = logm(np.block([
                                    [A_sindy_f10, B_sindy_f10],
                                    [np.zeros((B_sindy_f10.shape[1], B_sindy_f10.shape[0])), np.eye((B_sindy_f10.shape[1]))]
                                ]))/delta_t
            A_sindy_f10_c = phi_sindy_f10[:A_sindy_f10.shape[0], :A_sindy_f10.shape[1]]
            B_sindy_f10_c = phi_sindy_f10[:B_sindy_f10.shape[0], A_sindy_f10.shape[1]:]    
            
            X_k_norm = self.build_sindy_dyn_frcst_inputs(z1_k, interpolated_drivers, self.X_reg_norm_dict_sindy, self.pca_coupling, kp_idx, f10_idx, self.model_params,\
                normalization_method, input_features, k = int(k))   
            
            qF_norm = self.move_column(np.copy(X_k_norm).T, 5, self.sindy_tgt_col).T
            q0_norm_sindy = np.copy(qF_norm[-10:])
            
            
            solution_sindy = solve_ivp(
                self.ode_func_sindy,
                t_span,
                q0_norm_sindy.flatten(),
                args = (interpolated_drivers, A_sindy_joint_c, B_sindy_joint_c, A_sindy_f10_c, B_sindy_f10_c, self.sindy_tgt_col),
                method = 'RK45',
                t_eval = t_interval
            )
            
            t = self.sub_intervals*solution_sindy.t
            
            q_sol_sindy = np.full((len(q0_norm_sindy.flatten()), len(t_interval)), np.nan)
            q_sol_sindy[:, :len(solution_sindy.t)] = solution_sindy.y
            z_sindy = np.copy(q_sol_sindy * self.Y_reg_norm_dict_sindy['x_std'] + self.Y_reg_norm_dict_sindy['x_mean'])
            z_sindy = z_sindy[:, self.sub_intervals*self.delta_rho_ic:]
            self.z_results_lst.append(z_sindy)
        
        ###########################################dmd###############################################

        A_dmd = np.copy(np.copy(B_nl_dmd_discrete[:, 4:(-4)]))
        B_dmd = np.copy(np.concatenate([B_nl_dmd_discrete[:, :4], B_nl_dmd_discrete[:, (-4):]], axis = 1))

        phi_dmd = logm(np.block([
                                [A_dmd, B_dmd],
                                [np.zeros((B_dmd.shape[1], B_dmd.shape[0])), np.eye((B_dmd.shape[1]))]
                            ]))/delta_t
        A_dmd_c = phi_dmd[:A_dmd.shape[0], :A_dmd.shape[1]]
        B_dmd_c = phi_dmd[:B_dmd.shape[0], A_dmd.shape[1]:]


        X_k = np.concatenate([z1_k, interpolated_drivers[f10_idx:, k].reshape((-1, 1)), (interpolated_drivers[f10_idx, k] * interpolated_drivers[kp_idx, k]).reshape((-1, 1)), 
            (interpolated_drivers[kp_idx, k] * interpolated_drivers[kp_idx, k]).reshape((-1, 1))])
        X_k_norm = u.normalize_with_dict(X_k, self.X_reg_norm_dict_nl_dmd, normalization_method)   
        X_k_norm = np.concatenate([interpolated_drivers[1:f10_idx, k].reshape((-1, 1)), X_k_norm])

        q0_norm_dmd = np.copy(X_k_norm[4:(-4), :])

        solution_dmd  = solve_ivp(
            self.ode_func_dmd,
            t_span,
            q0_norm_dmd.flatten(),
            args = (interpolated_drivers, A_dmd_c, B_dmd_c),
            method = 'RK45',
            t_eval = t_interval
        )

        t = self.sub_intervals*solution_dmd.t
        q_sol_dmd = np.full((len(q0_norm_dmd.flatten()), len(t_interval)), np.nan)
        q_sol_dmd[:, :len(solution_dmd.t)] = solution_dmd .y
        z_dmd = np.copy(q_sol_dmd * self.Y_reg_norm_dict_nl_dmd['x_std'] + self.Y_reg_norm_dict_nl_dmd['x_mean'])
        z_dmd = z_dmd[:, self.sub_intervals*self.delta_rho_ic:]
        self.z_results_lst.append(z_dmd)
        self.t = t[:(-self.sub_intervals*self.delta_rho_ic)]
        self.z_dict = {}
        models = list(self.selected_bf_dict.keys()) + ['dmd']
        for k, z in enumerate(self.z_results_lst):
            self.z_dict[models[k]] = self.z_results_lst[k]


class rope_data_interpolator( PythonAtmosphere ):

    def __init__( self, data: rope_data, earth_shape: BodyShape = None, sigma_point_value: float = 0.0 ):
        super().__init__()

        self.data = data
        self.sigma_point_value = sigma_point_value
        self.j2000: Frame = FramesFactory.getEME2000()
        self.utc: UTCScale = TimeScalesFactory.getUTC()
        
        # if earth_shape is not None:
        #     self.shape = earth_shape
        # else:
        #     self.shape: BodyShape = OneAxisEllipsoid( Constants.IERS2010_EARTH_EQUATORIAL_RADIUS, Constants.IERS2010_EARTH_FLATTENING, FramesFactory.getITRF( IERSConventions.IERS_2010, False ) )


    def __compute_density__( self, lla: np.array, T: float ) -> tuple[float, float]:
        '''
        Density function matching WVU MATLAB implementation
        
        Inputs:
          lla: np.array of shape (3): lon (deg) / lat (deg) / alt (km)
          T: float: UTC fractional hour of day
        
        Outputs:
          tuple of atmospheric density output in kg/m^3 and uncertainty variance
        '''
        
        # try:
        # Read command-line arguments
        lla = lla.astype( float )
        T = float( T )  # Delta t in seconds
        point = np.array( [ lla[1], lla[0], lla[2] ] )  # (LAT, LT, ALT) ---> (LT, LAT, ALT)

        # Prepare interpolation variables
        t = self.data.t

        # Define grid for interpolation
        alt0 = 100  # Initial altitude in km
        step = 23.33333334  # Altitude step in km

        lt = np.arange( 0, self.data.U0.shape[0] )
        lat = np.linspace( -90, 81, self.data.U0.shape[1]) 
        alt = np.arange( alt0, alt0 + step * ( self.data.U0.shape[2] ), step )
        
        my_interpolating_U0 = rgi( (lt, lat, alt), self.data.U0, bounds_error=False, fill_value=None)
        my_interpolating_mu0 = rgi( (lt, lat, alt), self.data.mu0, bounds_error=False, fill_value=None) 
        
        # Perform interpolation
        U0_interp = my_interpolating_U0(point)
        mu0_interp = my_interpolating_mu0(point)

        sindy_interpolators_lst = []
        for sindy_model in list(self.data.z_dict.keys()):
            sindy_interpolators_lst.append(interp1d(t.flatten(), self.data.z_dict[sindy_model], kind='linear', axis=1, fill_value = "interpolate"))
        
        z_uncertainties_values_lst = [ 10 ** (U0_interp @ interpolator(T) + mu0_interp.reshape((-1, 1))) for interpolator in sindy_interpolators_lst]
        density_std = np.std(z_uncertainties_values_lst)
        density_mean = np.median(z_uncertainties_values_lst)
        
        return np.array([
            float(z_uncertainties_values_lst[0].item()), 
            float(z_uncertainties_values_lst[2].item()), 
            float(z_uncertainties_values_lst[4].item()), 
            float(z_uncertainties_values_lst[5].item()), 
            float(z_uncertainties_values_lst[6].item()), 
            float(density_mean.item()), 
            float(density_std.item())
        ])
    
    def interpolate_density_multi_rows( self, Tlla: np.array) -> tuple[float, float]:
        '''
        Function to inyterpolate density using multiple inputs
        
        Inputs:
          Tlla: np.array of shape (n, 3): T (sec) / lon (deg) / lat (deg) / alt (km)
          where n is the number of required inputs
        
        Outputs:
          tuple of atmospheric density output in kg/m^3 and uncertainty variance
        '''
        # print(Tlla)
        # try:
        # Read command-line arguments
        lla = Tlla[:, 1:]
        T = Tlla[:, 0]
        # print(T)
        points = np.column_stack([lla[:, 1], lla[:, 0], lla[:, 2]])  # (LAT, LT, ALT) ---> (LT, LAT, ALT)

        # Prepare interpolation variables
        t = self.data.t

        # Define grid for interpolation
        alt0 = 100  # Initial altitude in km
        step = 23.33333334  # Altitude step in km

        lt = np.arange( 0, self.data.U0.shape[0] )
        lat = np.linspace( -90, 81, self.data.U0.shape[1]) 
        alt = np.arange( alt0, alt0 + step * ( self.data.U0.shape[2] ), step )
        
        my_interpolating_U0 = rgi( (lt, lat, alt), self.data.U0, bounds_error=False, fill_value=None)
        my_interpolating_mu0 = rgi( (lt, lat, alt), self.data.mu0, bounds_error=False, fill_value=None) 
        
        # Perform interpolation
        U0_interp = my_interpolating_U0(points)
        mu0_interp = my_interpolating_mu0(points)

        sindy_interpolators_lst = []
        for sindy_model in list(self.data.z_dict.keys()):
            sindy_interpolators_lst.append(interp1d(t.flatten(), self.data.z_dict[sindy_model], kind='linear', axis=1, fill_value = "interpolate"))
        # print(T[-1], t.min(), t.max())
        z_uncertainties_values_lst = [ 10 ** (np.sum(U0_interp * interpolator(T).T, axis = 1).reshape((-1, 1)) + mu0_interp.T.reshape((-1, 1))) for interpolator in sindy_interpolators_lst]
        interpolated_values = np.stack(z_uncertainties_values_lst).squeeze(-1).T

        density_std = np.std(interpolated_values, axis = 1)
        density = np.median(interpolated_values, axis = 1)
        density_poly = interpolated_values[:, 0]
        density_poly_all = interpolated_values[:, 2]
        density_dmd = interpolated_values[:, 6]

        return density_poly, density_poly_all, density_dmd, density, density_std

    def interpolate( self, timestamps: np.array, lla: np.array, forward_propagation: int = 3) -> tuple[float, float]:
        '''
        Function to inyterpolate density using multiple inputs
        
        Inputs:
          timestamps: np.array of shape (n, 1): datetime64[ns]
          lla: np.array of shape (n, 3): T (sec) / lon (deg) / lat (deg) / alt (km)
          where n is the number of required inputs
        
        Outputs:
          tuple of atmospheric density output in kg/m^3 and uncertainty variance
        '''
        timestamps = pd.to_datetime(timestamps)
        if (self.data.date_series is None) | ~((np.all(timestamps >= self.data.date_series[0])) & (np.all(timestamps <= self.data.date_series[-1]))):    
            if np.ndim(timestamps) == 0 or (hasattr(timestamps, 'shape') and timestamps.shape == ()):
                print(f'System is propagating from {timestamps} for {forward_propagation} days')   
                self.data.propagate_models(pd.to_datetime(timestamps), forward_propagation = forward_propagation)

            else:
                print(f'System is propagating from {timestamps[0]} for {forward_propagation} days')   
                self.data.propagate_models(pd.to_datetime(timestamps[0]), forward_propagation = forward_propagation)

        t = self.data.t
        
        T = (timestamps - self.data.date_series[0]).total_seconds()
        # print(np.max(T), t.min(), t.max())
        
        points = np.column_stack([lla[:, 1], lla[:, 0], lla[:, 2]])  # (LAT, LT, ALT) ---> (LT, LAT, ALT)

        # Prepare interpolation variables
        # Define grid for interpolation
        alt0 = 100  # Initial altitude in km
        step = 23.33333334  # Altitude step in km

        lt = np.arange( 0, self.data.U0.shape[0] )
        lat = np.linspace( -90, 81, self.data.U0.shape[1]) 
        alt = np.arange( alt0, alt0 + step * ( self.data.U0.shape[2] ), step )
        
        my_interpolating_U0 = rgi( (lt, lat, alt), self.data.U0, bounds_error=False, fill_value=None)
        my_interpolating_mu0 = rgi( (lt, lat, alt), self.data.mu0, bounds_error=False, fill_value=None) 
        
        # Perform interpolation
        U0_interp = my_interpolating_U0(points)
        mu0_interp = my_interpolating_mu0(points)

        sindy_interpolators_lst = []
        for sindy_model in list(self.data.z_dict.keys()):
            sindy_interpolators_lst.append(interp1d(t.flatten(), self.data.z_dict[sindy_model], kind='linear', axis=1, fill_value = "interpolate"))
        
        z_uncertainties_values_lst = [ 10 ** (np.sum(U0_interp * interpolator(T).T, axis = 1).reshape((-1, 1)) + mu0_interp.T.reshape((-1, 1))) for interpolator in sindy_interpolators_lst]
        interpolated_values = np.stack(z_uncertainties_values_lst).squeeze(-1).T

        density_std = np.std(interpolated_values, axis = 1)
        density = np.mean(interpolated_values, axis = 1)
        density_poly = interpolated_values[:, 0]
        density_poly_all = interpolated_values[:, 2]
        density_dmd = interpolated_values[:, -1]

        return density_poly, density_poly_all, density_dmd, density, density_std
