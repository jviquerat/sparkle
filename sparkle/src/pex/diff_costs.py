from types import SimpleNamespace
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np

from sparkle.src.core.factory import Factory
from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.utils.default import set_default


###############################################
class DifferentiableCost:
    """
    Base class for differentiable cost functions
    """
    def __init__(self,
                 spaces: EnvSpaces,
                 pms: SimpleNamespace) -> None:
        self.spaces = spaces
        self.pms = pms

    def __call__(self, x: jnp.ndarray) -> float:
        """
        Computes the cost
        """
        raise NotImplementedError


###############################################
class MaximinCost(DifferentiableCost):
    """
    Maximize the minimum distance between points (maximin criterion)
    Implemented by minimizing the Coulomb-like potential energy: sum(1 / distance^2)
    """
    def __init__(self, spaces: EnvSpaces, pms: SimpleNamespace) -> None:
        super().__init__(spaces, pms)

        self.eps = set_default("eps", 1.0e-8, pms)

    # x shape: (N, D)
    def __call__(self, x: jnp.ndarray) -> float:
        n_points = x.shape[0]

        # Compute pairwise distances
        # diff: (N, N, D)
        diff = x[:, None, :] - x[None, :, :]

        # dist_sq: (N, N)
        dist_sq = jnp.sum(diff**2, axis=-1)

        # Mask out diagonal (distance to self is 0)
        # We add a large value to the diagonal so 1/d becomes 0
        mask_diag = jnp.eye(n_points, dtype=bool)
        dist_sq = jnp.where(mask_diag, jnp.inf, dist_sq)

        # Potential energy: sum(1 / dist^2)
        # 1/inf is 0, so diagonal contributes 0
        energy = jnp.sum(1.0 / (dist_sq + self.eps))

        return energy


###############################################
class MinimaxCost(DifferentiableCost):
    """
    Minimize the maximum distance from any point in the domain to the
    design (minimax criterion). Approximated by minimizing the LogSumExp
    of distances from a large Monte Carlo set to the nearest design point
    """
    def __init__(self, spaces: EnvSpaces, pms: SimpleNamespace) -> None:
        super().__init__(spaces, pms)

        self.n_samples = set_default("n_mc_samples", 1000, pms)
        self.alpha     = set_default("alpha", 50.0, pms)
        self.mc_set    = None
        self._generate_mc_set()

    def _generate_mc_set(self):
        """
        Generates a fixed Monte Carlo set for the domain approximation
        """
        self.mc_set = np.random.uniform(
            low=self.spaces.xmin,
            high=self.spaces.xmax,
            size=(self.n_samples, self.spaces.dim)
        )

    def __call__(self, x: jnp.ndarray) -> float:
        # mc_set is (M, D), x is (N, D)
        # Compute squared distances from MC set to design X
        # diff: (M, N, D)
        diff = self.mc_set[:, None, :] - x[None, :, :]
        dist_sq = jnp.sum(diff**2, axis=-1) # (M, N)

        # For each sample (M), find distance to closest design point
        min_dists = jnp.min(dist_sq, axis=1) # (M,)

        # Take the LogSumExp of these distances
        cost = (1.0 / self.alpha) * logsumexp(self.alpha * min_dists)

        return cost


###############################################
class MaxProCost(DifferentiableCost):
    """
    Minimizes the maximum projection criterion
    This criterion encourages points to be separated in all sub-projections
    by minimizing the sum of the inverse product of squared coordinate
    differences
    """
    def __init__(self, spaces: EnvSpaces, pms: SimpleNamespace) -> None:
        super().__init__(spaces, pms)

        self.eps = set_default("eps", 1.0e-8, pms)

    def __call__(self, x: jnp.ndarray) -> float:
        n_points = x.shape[0]

        # Compute pairwise coordinate differences
        # diff: (N, N, D)
        diff = x[:, None, :] - x[None, :, :]
        diff_sq = diff**2

        # Compute product of inverse squared differences over dimensions
        inv_diff_sq = 1.0 / (diff_sq + self.eps)

        # Product over dimensions k=1..d: (N, N)
        prod = jnp.prod(inv_diff_sq, axis=-1)

        # Mask out diagonal
        mask_diag = jnp.eye(n_points, dtype=bool)

        # Set diagonal to 0 so it doesn't contribute to sum
        prod = jnp.where(mask_diag, 0.0, prod)

        # Sum over all pairs
        cost = jnp.sum(prod)

        return cost


# Declare factory
cost_factory = Factory()

# Register costs
cost_factory.register("maximin", MaximinCost)
cost_factory.register("minimax", MinimaxCost)
cost_factory.register("maxpro",  MaxProCost)
