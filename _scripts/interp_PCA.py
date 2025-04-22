#%%=====

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree as kdt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


#%%======

#### testing function
def f(x,y,z,length_scales=[0.5, 0.5, 0.5], phases=[0.0, 0.0, 0.0]):
    L_x, L_y, L_z = length_scales
    ph_x, ph_y, ph_z = phases
    f1 = np.cos(2 * np.pi * x / L_x + ph_x)
    f2 = np.cos(2 * np.pi * y / L_y + ph_y)
    f3 = np.cos(2 * np.pi * z / L_z + ph_z) + np.cos(np.pi * z / L_z)
    return f1*f2*f3

def interp_knn(xn, data, tree, k=8):

    dist, indices = tree.query(xn, k=k)
    d = data[indices]
    w = 1 / dist
    return np.sum(w * d) / np.sum(w) # weighted knn sum

def get_proper_indices(ind, n_j):

    n, i ,j = 0, 0, 0
    while n != ind:
        if j < n_j:
            j += 1
        else:
            j = 0
            i += 1
        n = i * n_j + j

    return i, j

def interp_knn_pca(xn, mu, U, q, tree, scaler, k=8):

    dist, indices = tree.query(xn, k=k)
    X = scaler.transform((U.T @ q).reshape((1,-1)))[:,indices]
    w = 1 / dist
    return np.sum(w * X) / np.sum(w) # weighted knn sum

# def interp_trilinear(xn, xp, data, tree):

#     _, ind = tree.query(xn, k=8)
#     xg = xp[ind,:]

#     return None

#%%=========

#### build mesh
x = np.linspace(0, 1, 50)
z = np.linspace(0, 1, 51)
mesh = np.meshgrid(x, x, z)
mesh_2D = np.meshgrid(x, x)
xg = np.column_stack([ax.ravel() for ax in mesh])
xg_2D = np.column_stack([ax.ravel() for ax in mesh_2D])
tree = kdt(xg)
tree_2D = kdt(xg_2D)

D = np.zeros(mesh[0].shape)
D_2D = np.zeros((len(x)**2, len(z)))
for i, _x in enumerate(x):
    for j, _y in enumerate(x):
        for k, _z in enumerate(z):
            _f =  f(_x, _y, _z)
            D_2D[i*len(x)+j,k] = _f
            D[i,j,k] = _f
Dg = D.ravel()

scaler = StandardScaler()
scaler.fit(D_2D.T)
scaled_data = scaler.transform(D_2D.T)
pca = PCA(n_components=2)
pca.fit(scaled_data)
xform_data = pca.transform(scaled_data)
U = pca.components_


#%%========
Ntest = 1000
vals = np.zeros(Ntest)
itp = np.zeros(Ntest)
itp_pca = np.zeros(Ntest)
rand = np.random.random(size=(Ntest, 3))
rand[:,2] = 0
for i in range(Ntest):
    vals[i] = f(rand[i,0], rand[i,1], rand[i,2])
    itp[i] = interp_knn(rand[i,:], Dg, tree, k=8)
    itp_pca[i] = interp_knn_pca(rand[i,:2],
                                scaler.mean_, U, xform_data[0, :],
                                tree_2D, scaler, k=4)

err = np.abs(vals - itp)
err_pca = np.abs(vals - itp_pca)
plt.hist(err, bins=20, alpha=0.5)
#plt.hist(err_pca, bins=20, alpha=0.5)
plt.show()

# %% 
