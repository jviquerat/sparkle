# Generic imports
import numpy as np

# Compute distance between two sets of coordinates
def distance(xi, xj):

    return np.linalg.norm(xi - xj)

# Compute nearest neighbour of one point within vector
def nearest_one_to_all(x, i):

    n_points = x.shape[0]
    d_min    = 1.0e8
    p_min    =-1

    for j in range(n_points):
        if (i==j): continue

        d = distance(x[i], x[j])
        if (d < d_min):
            d_min = d
            p_min = j

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

    n_points = x.shape[0]
    dmin     = 1.0e8

    for i in range(n_points):
        for j in range(i+1, n_points):
            dist = distance(x[i], x[j])
            if (dist < dmin): dmin = dist

    return dmin
