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
    # candidate_mask is a mask for candidate points (true if point is candidate)
    selected = np.zeros(n_target, dtype=int)
    candidate_mask = np.ones(n_points, dtype=bool)

    # Array of minimal squared distance from each point to any selected point
    # It is initialized with np.inf and updated after each point selection
    # We use squared distances for efficiency
    min_sq_dists = np.full(n_points, np.inf)

    # Select the first point randomly
    k = np.random.randint(0, n_points)
    selected[0] = k
    candidate_mask[k] = False
    n_selected = 1

    # Compute squared distances from the first point to all other candidates
    # The mask selects only candidates points
    diffs = x[candidate_mask] - x[k]
    sq_dists_to_last = np.sum(diffs*diffs, axis=1)

    # Update minimal distances for candidates
    min_sq_dists[candidate_mask] = sq_dists_to_last

    # Loop until n_target points are selected
    while n_selected < n_target:

        # Get min distances for current candidates
        current_candidate_min_sq_dists = min_sq_dists[candidate_mask]

        # Find position (relative index) of the farthest candidate
        farthest_candidate_pos = np.argmax(current_candidate_min_sq_dists)

        # Get the original index of the farthest candidate
        candidate_original_indices = np.where(candidate_mask)[0]
        farthest_original_idx = candidate_original_indices[farthest_candidate_pos]

        # Add the farthest point to the selected set
        selected[n_selected] = farthest_original_idx
        candidate_mask[farthest_original_idx] = False # Mark as selected
        n_selected += 1

        # Exit if done
        if n_selected == n_target:
            break

        # Update min distances for remaining candidates
        last_selected_coords = x[farthest_original_idx]

        # Calculate sq distances from the *newly added* point to remaining candidates
        diffs = x[candidate_mask] - last_selected_coords
        sq_dists_to_last = np.sum(diffs * diffs, axis=1)

        # Update min distances: take element-wise min of current and new distances
        min_sq_dists[candidate_mask] = np.minimum(min_sq_dists[candidate_mask],
                                                  sq_dists_to_last)

    # Return the subset of points corresponding to the selected indices
    return x[selected]
