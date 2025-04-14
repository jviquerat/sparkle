import math

import numpy as np

from sparkle.src.utils.compare import compare
from sparkle.src.utils.distances import (
    distance,
    pairwise_distances,
    min_distance_in_set,
    min_max_distance_in_set,
    nearest_neighbors_in_set,
    nearest_neighbor_in_set,
)


###############################################
def test_distance():

    x = np.zeros(5)
    y = np.ones(5)
    d = distance(x, y)
    assert compare(d, math.sqrt(5.0), 1.0e-15)

###############################################
def test_pairwise_distances():

    x = np.array([[1.0, 1.0],
                  [2.0, 2.0],
                  [4.0, 4.0]])
    d = pairwise_distances(x, x)
    ref = np.array([[0.,         1.41421356, 4.24264069],
                    [1.41421356, 0.,         2.82842712],
                    [4.24264069, 2.82842712, 0.        ]])
    assert np.allclose(d, ref)

###############################################
def test_nearest_neighbor_in_set():

    x = np.array([[1.0, 1.0],
                  [2.0, 2.0],
                  [4.0, 4.0]])
    d, p = nearest_neighbor_in_set(x, 2)
    assert compare(d, math.sqrt(8.0), 1.0e-15)
    assert p == 1

###############################################
def test_nearest_neighbors_in_set():

    x = np.array([[1.0, 1.0],
                  [2.0, 2.0],
                  [4.0, 4.0]])
    d, p = nearest_neighbors_in_set(x)
    assert np.allclose(d, [math.sqrt(2.0), math.sqrt(2.0), math.sqrt(8.0)])
    assert np.allclose(p, [1,0,1])

###############################################
def test_min_distance_in_set():

    x = np.array([[1.0, 1.0],
                  [2.0, 2.0],
                  [4.0, 4.0]])
    d = min_distance_in_set(x)
    assert compare(d, math.sqrt(2.0), 1.0e-15)

###############################################
def test_min_max_distance_in_set():

    x = np.array([[1.0, 1.0],
                  [2.0, 2.0],
                  [4.0, 4.0]])
    dmin, dmax = min_max_distance_in_set(x)
    assert compare(dmin, math.sqrt(2.0),  1.0e-15)
    assert compare(dmax, math.sqrt(18.0), 1.0e-15)
