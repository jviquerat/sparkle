# Generic imports
import torch
import numpy as np

# Compute distance between two sets of coordinates
def distance(xi, xj):

    return np.linalg.norm(xi - xj)

# Compute distances from all points of xi to all points of xj
# xi and xj have shapes (n_batch, dim)
def distance_all_to_all(xi, xj):

    return np.linalg.norm(xi[:,np.newaxis,:] - xj[np.newaxis,:,:], axis=-1)

# Compute nearest neighbour of one point within vector
def nearest_one_to_all(x, i):

    # Compute norm of x-x_i
    d = np.linalg.norm(x - x[i], axis=1)

    # Set ith component to infinity
    d[i] = np.inf

    # Create masked array and find min distance
    p_min = d.argmin()
    d_min = d[p_min]

    return d_min, p_min

# Compute nearest neighbour for all input coordinates
def nearest_all_to_all(x):

    d = distance_all_to_all(x, x)
    np.fill_diagonal(d, np.inf)
    p_nearest = d.argmin(axis=1)
    d_nearest = d[np.arange(len(x)), p_nearest]

    return d_nearest, p_nearest

# Compute minimal distance between two points within vector
def min_distance(x):

    d, p = nearest_all_to_all(x)
    dmin = np.min(d)

    return dmin

# Compute minimal and maximal distances between two points within vector
def min_max_distance(x):

    n = x.shape[0]
    d = distance_all_to_all(x,x)

    mask = np.zeros((n,n))
    np.fill_diagonal(mask, 1)

    dm   = np.ma.masked_array(d, mask)
    dmin = dm.min()
    dmax = dm.max()

    return dmin, dmax

# Distance between two torch tensors
def tensor_distance(x, y):

    return torch.linalg.vector_norm(x-y)
