import numpy as np
from numpy import ndarray

from sparkle.src.utils.error import error

###############################################
### Furthest point sampling
### x: array of point coordinates of shape (n_points, dim)
### n_points: target number of points
def FPS(x: ndarray, n_points: int) -> ndarray:

    if (x.shape[0] < n_points):
        error("fps", "fps",
              "Input list has less than required number of points")

    if (x.shape[0] == n_points):
        return np.array(x)

    # Work on indices list for better efficiency
    lst      = np.arange(x.shape[0]).tolist()
    k        = np.random.randint(0, x.shape[0])
    selected = [k]
    lst.pop(k)

    while (len(selected) < n_points):
        distances = np.zeros((len(lst), len(selected)))
        for j in range(len(selected)):
            distances[:,j] = np.linalg.norm(x[lst] - x[selected[j]], axis=1)

        min_dists = np.min(distances, axis=1)
        k         = np.argmax(min_dists)

        selected.append(lst[k])
        lst.pop(k)


    return x[selected]
