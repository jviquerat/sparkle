from typing import Tuple, Union

import numpy as np
import torch
from numpy import ndarray


# Compute distance between two sets of coordinates
def distance(xi: ndarray, xj: ndarray) -> float:

    return np.linalg.norm(xi - xj)

# Compute distances from all points of xi to all points of xj
# xi and xj have shapes (n_batch, dim)
def distance_all_to_all(xi: ndarray, xj: ndarray) -> ndarray:

    return np.linalg.norm(xi[:,np.newaxis,:] - xj[np.newaxis,:,:], axis=-1)

# Compute nearest neighbour of one point within vector
def nearest_one_to_all(x: ndarray, i: int) -> Tuple[float, int]:

    # Compute norm of x-x_i
    d = np.linalg.norm(x - x[i], axis=1)

    # Set ith component to infinity
    d[i] = np.inf

    # Create masked array and find min distance
    p_min = d.argmin()
    d_min = d[p_min]

    return d_min, p_min

# Compute nearest neighbour for all input coordinates
def nearest_all_to_all(x: ndarray) -> Tuple[ndarray, ndarray]:

    d = distance_all_to_all(x, x)
    np.fill_diagonal(d, np.inf)
    p_nearest = d.argmin(axis=1)
    d_nearest = d[np.arange(len(x)), p_nearest]

    return d_nearest, p_nearest

# Compute minimal distance between two points within vector
def min_distance(x: ndarray) -> float:

    d, p = nearest_all_to_all(x)
    dmin = np.min(d)

    return dmin

# Compute minimal and maximal distances between two points within vector
def min_max_distance(x: ndarray) -> Tuple[float, float]:

    d = distance_all_to_all(x, x)
    np.fill_diagonal(d, np.inf)
    dmin = np.min(d)
    np.fill_diagonal(d, -np.inf)
    dmax = np.max(d)

    return dmin, dmax

# Distance between two torch tensors
def tensor_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    return torch.linalg.vector_norm(x-y)
