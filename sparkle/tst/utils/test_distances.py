# Generic imports
import math
import numpy as np

# Custom imports
from sparkle.src.utils.compare   import compare
from sparkle.src.utils.distances import distance, min_distance, min_max_distance, nearest_one_to_all, nearest_all_to_all, distance_all_to_all

###############################################
### Test distance
def test_distance():

    x = np.zeros(5)
    y = np.ones(5)
    d = distance(x, y)
    assert compare(d, math.sqrt(5.0), 1.0e-15)

###############################################
### Test distance_one_to_all
def test_distance_all_to_all():

    x = np.array([[1.0, 1.0],
                  [2.0, 2.0],
                  [4.0, 4.0]])
    d = distance_all_to_all(x, x)
    ref = np.array([[0.,         1.41421356, 4.24264069],
                    [1.41421356, 0.,         2.82842712],
                    [4.24264069, 2.82842712, 0.        ]])
    assert np.allclose(d, ref)

###############################################
### Test nearest_one_to_all
def test_nearest_one_to_all():

    x = np.array([[1.0, 1.0],
                  [2.0, 2.0],
                  [4.0, 4.0]])
    d, p = nearest_one_to_all(x, 2)
    assert compare(d, math.sqrt(8.0), 1.0e-15)
    assert p == 1

###############################################
### Test nearest_all_to_all
def test_nearest_all_to_all():

    x = np.array([[1.0, 1.0],
                  [2.0, 2.0],
                  [4.0, 4.0]])
    d, p = nearest_all_to_all(x)
    assert np.allclose(d, [math.sqrt(2.0), math.sqrt(2.0), math.sqrt(8.0)])
    assert np.allclose(p, [1,0,1])

###############################################
### Test min_distance
def test_min_distance():

    x = np.array([[1.0, 1.0],
                  [2.0, 2.0],
                  [4.0, 4.0]])
    d = min_distance(x)
    assert compare(d, math.sqrt(2.0), 1.0e-15)

###############################################
### Test min_max_distance
def test_min_max_distance():

    x = np.array([[1.0, 1.0],
                  [2.0, 2.0],
                  [4.0, 4.0]])
    dmin, dmax = min_max_distance(x)
    assert compare(dmin, math.sqrt(2.0),  1.0e-15)
    assert compare(dmax, math.sqrt(18.0), 1.0e-15)
