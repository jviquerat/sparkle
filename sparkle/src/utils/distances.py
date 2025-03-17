# Generic imports
import numpy as np

# Compute distance between two sets of coordinates
def distance(xi, xj):

    return np.linalg.norm(xi - xj)

# Compute nearest neighbour of one point within vector
def nearest_one_to_all(x, i):

    # Compute norm of x-x_i
    d = np.linalg.norm(x - x[i], axis=1)

    # Create mask for ith point
    mask    = np.zeros(x.shape[0])
    mask[i] = 1

    # Create masked array and find min distance
    dm    = np.ma.masked_array(d, mask)
    p_min = dm.argmin()
    d_min = d[p_min]

    return d_min, p_min

# Compute nearest neighbour for all input coordinates
def nearest_all_to_all(x):

    n_points  = x.shape[0]
    d_nearest = np.zeros(n_points)
    p_nearest = np.zeros(n_points, dtype=int)

    for i in range(n_points):
        d_min, p_min = nearest_one_to_all(x, i)
        d_nearest[i] = d_min
        p_nearest[i] = p_min

    return d_nearest, p_nearest

# Compute minimal distance between two points within vector
def min_distance(x):

    d, p = nearest_all_to_all(x)
    dmin = np.min(d)

    return dmin
