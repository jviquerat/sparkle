# Generic imports
import numpy as np

# Custom imports
from sparkle.src.utils.distances import distance, min_distance
from sparkle.src.utils.error     import error

###############################################
### Furthest point sampling
### x: list of point coordinates
### n_points: target number of points
def fps(x, n_points):

    if (len(x) < n_points):
        error("fps", "fps",
              "Input list has less than required number of points")

    if (len(x) == n_points):
        return np.array(x)

    k = np.random.randint(0, len(x))
    selected = [x[k]]
    x.pop(k)

    while (len(selected) < n_points):
        distances = np.zeros((len(x), len(selected)))
        for i in range(len(x)):
            for j in range(len(selected)):
                distances[i,j] = distance(x[i], selected[j])

        min_dists = np.min(distances, axis=1)
        k         = np.argmax(min_dists)

        selected.append(x[k])
        x.pop(k)

    return np.array(selected)
