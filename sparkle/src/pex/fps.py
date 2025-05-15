import numpy as np
from numpy import ndarray

from sparkle.src.utils.error import error


###############################################
def FPS(x: ndarray, n_target: int) -> ndarray:
    """
    Selects a subset of points using the Furthest Point Sampling (FPS) algorithm
    Uses an efficient incremental update approach with squared Euclidean distances

    Args:
        x: numpy array of shape (n, d)
        n_target: target nb of points to select

    Returns:
        a numpy array of shape (n_target, d) containing the selected points
    """
    # Check x shape
    if x.ndim != 2:
        error("fps", "fps",
              f"Input x must be of shape (n, d), but got shape {x.shape}")

    # Check for non-finite values (NaN or Inf) in the input array
    if not np.isfinite(x).all():
        error("fps", "fps",
              "Input array x contains non-finite values (NaN or Inf)")

    # Check for number of input points
    n_points = x.shape[0]
    if n_target > n_points:
        error("fps", "fps", f"n_points is lower than n_target")

    if n_target == n_points:
        return x.copy()

    # selected is the array of selected points indices
    # mask[p] = True if point p is candidate
    selected = np.zeros(n_target, dtype=int)
    mask     = np.ones(n_points, dtype=bool)

    # Array of minimal (squared) distance from each point to any selected point
    # It is initialized with np.inf and updated after each point selection
    min_dists = np.full(n_points, np.inf)

    # Select the first point randomly
    k           = np.random.randint(0, n_points)
    selected[0] = k
    mask[k]     = False
    n_selected  = 1

    # Compute (squared) distances from current point to candidates
    diff         = x[mask] - x[k]
    dists_to_crt = np.sum(diff*diff, axis=1)

    # Update minimal distances for candidates
    min_dists[mask] = dists_to_crt

    # Loop until n_target points are selected
    while n_selected < n_target:

        # Find index of farthest candidate *within the masked array*,
        # then retrieve its index in the original numbering
        farthest_idx     = np.argmax(min_dists[mask])
        original_indices = np.where(mask)[0]
        farthest_idx     = original_indices[farthest_idx]

        # Add the farthest point to the selected set
        selected[n_selected] = farthest_idx
        mask[farthest_idx]   = False # Mark as selected
        n_selected          += 1

        # Exit if done
        if n_selected == n_target:
            break

        # Calculate sq distances from the *newly added* point to remaining candidates
        diff         = x[mask] - x[farthest_idx]
        dists_to_crt = np.sum(diff*diff, axis=1)

        # Update min distances: take element-wise min of current and new distances
        min_dists[mask] = np.minimum(min_dists[mask], dists_to_crt)

    # Return the subset of points corresponding to the selected indices
    return x[selected]
