#%%============================== Imports ==============================%%#

#### adds parent path to PYTHONPATH for import
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#### basic functionality
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mproc

#### forecasting library
import forecast.models as md
import forecast.statevars as sv
from forecast.forecast import forecast

#%%============================== Basic Test ==============================%%#

t = np.linspace(0, 1, 1001)
t_hs = np.linspace(0, 1, 2*len(t)-1)

q0 = np.ones(10) # 10-fold variable space
u = np.zeros((len(t), 3)) # 3-fold driver space

u[:,0] = 0.1 * np.cos(2 * np.pi * t)
u[:,1] = 0.1 * np.cos(2 * np.pi * t + np.pi/2)
u[:,2] = 0.1 * np.cos(np.pi * t)**2 # kp stand-in

funcs, names = md._construct_linear_basis_SINDYc(q0, u[0,:])

# Xi = [np.random.random((len(q0), len(funcs))) for i in range(2)]
Xi = [np.zeros((len(q0), len(funcs))) for i in range(2)] 
for i in range(2):
    for j in range(len(q0)):
        n_ij = np.random.randint(1,3,2)
        Xi[0][j,j] = -0.1 * np.random.random(1) #(-1)**n_ij[0] # identity matrix
       # Xi[1][j,j] = -10 * np.random.random(1) #(-1)**n_ij[0]
Xi[0][:,len(q0):] = 2 * np.random.random(size=(len(q0), u.shape[-1])) - 1
Xi[1][:,len(q0):] = 2 * np.random.random(size=(len(q0), u.shape[-1])) - 1
Xi[1] = Xi[0].copy()

state = sv.SindycState("test", Xi, funcs, names,
                       q0, {"kp": 2}, "constant", 0.01)

job_params = {"drivers": u, 
              "solver_times_full": t, 
              "solver_times_half": t_hs,
              "driver_times": t,
              "forecast_times": None} # skips forecast interpolation

job = sv.TaskState(0, job_params, state)

res = md.SINDYc_forecast(job)

fig, ax = plt.subplots(2,1,figsize=(12.0,18.0))
for i in range(u.shape[-1]):
    ax[0].plot(t, u[:,i], lw=3)
for i in range(len(q0)):
    ax[1].plot(t, res[:,i])
plt.show()

#%%============================== Parallel Test ==============================%%#

N = 2
jobs = []
for i in range(N):
    task = sv.TaskState(i, job_params, state)
    jobs.append(task)

if __name__=="main":
    pool = mproc.Pool(4)
    res = forecast(jobs, pool=pool, verbose=True)

# %%
