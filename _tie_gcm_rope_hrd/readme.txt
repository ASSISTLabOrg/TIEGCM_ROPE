# TIE-GCM-ROPE usage instructions and brief description

-works at altitude limited to 450 km. This will be extended to 1000 km by the final delivery of phase 1.
-Initial condition uses a simple database classified using kp and f10.7 bins 
(this will eventually be replaced with a nowcast achieved using data assimilation)
- Current version uses the driver file from Celestrack. In the future, the most up to date Celestrack file 
will be downloaded automatically. Additionally, an option for the user to proivide the driver inputs will be 
enabled for forecast mode.

Some backend information
Once thye user provides initial time, position and number of days for the forward propagation, the system 
propagates the grid in time for the required amount of time. 
The initial conditions to the propagation are built using a classification for the initial vectors based on 
kp and f10.7 drivers table.
The propagation begins 6 hours before the date specified by the user to make sure that the system aligns with 
the external drivers by the time entered by the user.
The user then adds latitude (in degrees), local time (in hours) and altitudes (in km) at which to interpolate 
the thermospheric grid. 

##############################################Step by step installation and setup#################################
The following instructions will make the tool work unless there are system-wide settings that need to be fixed. 
To install the environment and run the tool, you need to run from terminal the following command:

1) install and manage all the required conda settings that may depend on your system/environment 
2-a) Run from terminal the command: conda env create -f ./tie_gcm_rope_env.yml
2-b) Open the file tie_gcm_rope_wrapper.ipynb and set up the conda environment to be tiegcm_rope_env.
    Notice that the setup of the conda environment depends on your framework. For instance in vscode you can 
    select the environment from the top right corner of your screen.
3) Start running the file tie_gcm_rope_wrapper.ipynb block by block:
    The first block is serves to import libraries
    The second block is the TIE-GCM ROPE tool and you can run with the provided demo inputs



######################################################################################################################
Before running everithing, when you modify the code, make sure that orekit-data.zip file is available in the current folder and is initialized 
through the code
vm = orekit.initVM()
setup_orekit_curdir( './' )

###############################################TIE-GCM-ROPE inputs description##########################################
latitude range : [-90, 90] deg
local time range: [0, 24] hours
altitude range: [100, 450] km
timestamps is an array of strings representing date-time in the format 'YYYY-mm-dd HH:MM:SS'


The inputs that the user must provide to use the emulator are the datetime or datetimes inputs in the from

'2003-11-03 00:00:00'

or if there is more than 1 timestamp they must be provided as an array for instance as

timestamps = np.array(['2003-11-03 00:00:00', '2003-11-03 10:00:00'])

At the same time the set of latitudes, local times and altitudes must be provided as an (n, 3) numpy.array.
For instance you can provide 

init_date_str = '2003-11-03 00:00:00' #which always signals the initial propagation date
lla_array = np.array([[87., 23.7, 440.5]])
forward_propagation = 6

as they are specified in the example, and the outputs will be 

{'density': array([1.93864391e-12]), 'standard deviation': array([7.20897136e-14])}

Another example with more than one input is 

init_date_str = '2003-11-03 00:00:00' #which always signals the initial propagation date
timestamps = np.array(['2003-11-03 04:00:00', '2003-11-03 10:00:00'])
lla_array = np.array([[87., 23.7, 440.5], [87., 23.7, 440.5]])
forward_propagation = 6

whose output is 

{'density': array([2.01949620e-12, 1.81244489e-12]), 'standard deviation': array([8.14883348e-14, 1.10739710e-13])}

These demos are intended to make the functioning of the tool as clear as possible.
##################################################################################################################
The propagation occurs at the row 

sindy.propagate_models(init_date = init_date, forward_propagation = forward_propagation)

Following, the interpolator is defined at the row

rope_density = rope.rope_data_interpolator( data = sindy )

by feeding it with the sindy object containing the propagated variables.

The interpolated density is calculated at the row 

density_poly, density_poly_all, density_dmd, density, density_std = rope_density.interpolate(timestamps, lla_array)

where density and density_std are the ensemble mean and the ensemble uncertainty. The other outputs are specific
densities corresponding to particular basis functions or other models. If they are of no interest one can just 
cover those outputs by using 

_, _, _, density, density_std = rope_density.interpolate(timestamps, lla_array)

