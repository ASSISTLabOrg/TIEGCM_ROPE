# ROPE Framework

**R**educed **O**rder **P**robabilistic **E**mulator framework.  
This repo contains python scripts for predicing the neutral density in the thermosphere using surrogate ML models trained on physics simulations (TIE-GCM, WAM-IPE).

## Contents
hrd_20250608 contains the latest Python version. Works at altitudes between 100 km and 980 km, but can extrapolate outside the boundaries. Refer to the demo.py file for applications. 
Initial conditions are calculated using the history of kp and f10.7. Latent space variables are calculated as a function of this initial conditions database. 
The initial conditions to the propagation are built using a classification for the initial vectors based on a kp and f10.7 drivers table. The propagation is suggested to be set to begin 6 hours before the date specified by the user to let the system align with the actual dynamics. There are howeve no meaningful differences if the initial delay is set to zero and the output is debiased.
### forecast/  
Contains the methods needed to use trained forecasting models to predict thermosphere neutral densities.
### train/  
Contains the methods used in training and re-training the various forecaster methods.
### _data/
Contains the data needed to run forecaster models. Temporary - these data will eventually be held at a remote location.
### _notebooks/  
Contains notebooks for testing and developing new models.
### _scripts/
Contains scripts for testing and developing new models.
