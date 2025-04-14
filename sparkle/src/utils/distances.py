from typing import Tuple

import numpy as np
import torch
from numpy import ndarray


def distance(xi: ndarray, xj: ndarray) -> float:
    """
    Computes the Euclidean distance between two points.

    Args:
        xi: The first point as a NumPy array.
        xj: The second point as a NumPy array.

    Returns:
        The Euclidean distance between xi and xj.
    """

    return np.linalg.norm(xi - xj)

def distance_all_to_all(xi: ndarray, xj: ndarray) -> ndarray:
    """
    Computes the Euclidean distances between all pairs of points from two sets.

    Args:
        xi: The first set of points as a NumPy array of shape (n_batch_i, dim).
        xj: The second set of points as a NumPy array of shape (n_batch_j, dim).

    Returns:
        A NumPy array of shape (n_batch_i, n_batch_j) containing the distances
        between all pairs of points.
    """

    return np.linalg.norm(xi[:,np.newaxis,:] - xj[np.newaxis,:,:], axis=-1)

def nearest_one_to_all(x: ndarray, i: int) -> Tuple[float, int]:
    """
    Computes the nearest neighbor of a point within a set of points.

    Args:
        x: The set of points as a NumPy array of shape (n_points, dim).
        i: The index of the point for which to find the nearest neighbor.

    Returns:
        A tuple containing:
            - The distance to the nearest neighbor (float).
            - The index of the nearest neighbor (int).
    """

    # Compute norm of x-x_i
    d = np.linalg.norm(x - x[i], axis=1)

    # Set ith component to infinity
    d[i] = np.inf

    # Create masked array and find min distance
    p_min = d.argmin()
    d_min = d[p_min]

    return d_min, p_min

def nearest_all_to_all(x: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Computes the nearest neighbor for each point in a set of points.

    Args:
        x: The set of points as a NumPy array of shape (n_points, dim).

    Returns:
        A tuple containing:
            - A NumPy array of shape (n_points,) containing the distances to
              the nearest neighbors.
            - A NumPy array of shape (n_points,) containing the indices of
              the nearest neighbors.
    """

    d = distance_all_to_all(x, x)
    np.fill_diagonal(d, np.inf)
    p_nearest = d.argmin(axis=1)
    d_nearest = d[np.arange(len(x)), p_nearest]

    return d_nearest, p_nearest

def min_distance(x: ndarray) -> float:
    """
    Computes the minimum distance between any two points in a set.

    Args:
        x: The set of points as a NumPy array of shape (n_points, dim).

    Returns:
        The minimum distance between any two points in x.
    """

    d, p = nearest_all_to_all(x)
    dmin = np.min(d)

    return dmin

def min_max_distance(x: ndarray) -> Tuple[float, float]:
    """
    Computes the minimum and maximum distances between any two points in a set.

    Args:
        x: The set of points as a NumPy array of shape (n_points, dim).

    Returns:
        A tuple containing:
            - The minimum distance between any two points in x.
            - The maximum distance between any two points in x.
    """

    d = distance_all_to_all(x, x)
    np.fill_diagonal(d, np.inf)
    dmin = np.min(d)
    np.fill_diagonal(d, -np.inf)
    dmax = np.max(d)

    return dmin, dmax

def tensor_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the Euclidean distance between two PyTorch tensors.

    Args:
        x: The first tensor.
        y: The second tensor.

    Returns:
        The Euclidean distance between x and y.
    """

    return torch.linalg.vector_norm(x-y)
