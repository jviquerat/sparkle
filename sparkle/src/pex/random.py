from types import SimpleNamespace

import numpy as np

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.pex.base import BasePex
from sparkle.src.pex.fps import FPS
from sparkle.src.utils.default import set_default


###############################################
class Random(BasePex):
    """
    Random experiment plan.

    This class implements a simple random sampling method for generating
    experiment plans. Points are uniformly distributed within the search space.
    """
    def __init__(self, spaces: EnvSpaces, pms: SimpleNamespace) -> None:
        """
        Initializes the Random experiment plan.

        Args:
            spaces: The environment's search space definition.
            pms: A SimpleNamespace object containing parameters for the experiment plan.
        """
        super().__init__(spaces, pms)

        self.name = "random"

        self.reset()

    def reset(self) -> None:
        """
        Resets the Random experiment plan by generating new sample points.

        This method generates a new set of points randomly distributed
        within the search space.
        """

        self.x_ = np.random.uniform(low  = self.xmin,
                                    high = self.xmax,
                                    size = (self.n_points_, self.dim))

###############################################
class RFPS(BasePex):
    """
    Random experiment plan with Furthest Point Sampling (FPS).

    This class implements a random sampling method followed by a Furthest
    Point Sampling step to improve the distribution of points.
    """
    def __init__(self, spaces: EnvSpaces, pms: SimpleNamespace) -> None:
        """
        Initializes the RandomFPS experiment plan.

        Args:
            spaces: The environment's search space definition.
            pms: A SimpleNamespace object containing parameters for the experiment plan,
                including the factor for oversampling (factor).
        """
        super().__init__(spaces, pms)

        self.name     = "random_fps"
        self.factor   = set_default("factor", 10, pms)
        self.n_target = self.factor*self.n_points_

        self.reset()

    def reset(self) -> None:
        """
        Resets the RandomFPS experiment plan by generating new sample points.

        This method generates an initial set of random points, then applies
        Furthest Point Sampling to select a subset of these points that are
        well-distributed.
        """

        # x = np.random.uniform(low  = self.xmin,
        #                       high = self.xmax,
        #                       size = (self.n_target, self.dim))

        x = generate_halton_sequence(self.n_target, self.dim, skip_points=5000)

        for i in range(self.n_target):
            x[i] = self.xmin + x[i]*(self.xmax - self.xmin)

        self.x_ = FPS(x, self.n_points_)

        best_minimax = self.minimax()
        for k in range(500):
            p_selected  = np.random.choice(self.n_points_, size=1)
            p_candidate = np.random.choice(self.n_target, size=1)

            old_x = self.x_[p_selected].copy()
            self.x_[p_selected] = x[p_candidate]
            minimax = self.minimax()
            if minimax < best_minimax:
                best_minimax = minimax
            else:
                self.x_[p_selected] = old_x

# You'll need a list of prime numbers. For simplicity, here are the first few.
# For higher dimensions, you'd need a more extensive list or a prime generator.
FIRST_PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
    157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229
] # First 50 primes, good for up to d=50

def _van_der_corput_point(index: int, base: int) -> float:
    """
    Calculates a single Van der Corput point for a given index and base.
    Uses 1-based indexing for the sequence generation.
    """
    result = 0.0
    f = 1.0
    i_temp = index # Use the provided index directly
    while i_temp > 0:
        f /= base
        result += (i_temp % base) * f
        i_temp //= base
    return result

def generate_halton_sequence(num_points: int, num_dimensions: int, skip_points: int = 0) -> np.ndarray:
    """
    Generates a Halton sequence of points in the unit hypercube [0,1)^d.

    Args:
        num_points: The number of Halton points to generate.
        num_dimensions: The dimensionality of the points.
        skip_points: Number of initial points in the sequence to skip (burn-in).
                     Skipping points (e.g., a few hundred or thousand) can
                     improve properties by avoiding correlations in early sequence values.

    Returns:
        A NumPy array of shape (num_points, num_dimensions) containing
        the Halton sequence. Returns an empty array if num_dimensions
        is too large for the pre-defined primes list.
    """
    if num_dimensions <= 0:
        return np.empty((num_points, 0))
    if num_dimensions > len(FIRST_PRIMES):
        print(f"Warning: Requested {num_dimensions} dimensions, but only {len(FIRST_PRIMES)} primes are defined. "
              f"Returning empty array or consider adding more primes.")
        # Or raise ValueError
        return np.empty((num_points, num_dimensions)) # Or an empty array based on how you want to handle error

    halton_sequence = np.zeros((num_points, num_dimensions))
    
    prime_bases = FIRST_PRIMES[:num_dimensions]

    for i in range(num_points):
        sequence_index = i + 1 + skip_points # Ensure 1-based index for VDC, and apply skip
        for j in range(num_dimensions):
            halton_sequence[i, j] = _van_der_corput_point(sequence_index, prime_bases[j])
            
    return halton_sequence
