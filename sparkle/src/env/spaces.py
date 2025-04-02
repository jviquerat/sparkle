from types import SimpleNamespace
from typing import Any, List, Optional

from numpy import ndarray


###############################################
class EnvSpaces:
    """
    A class holding information about the search space.

    This class stores and manages information about the search space,
    including its dimensionality, bounds, and other relevant properties.
    """
    def __init__(self,
                 spaces: Any,
                 pms: Optional[SimpleNamespace]=None) -> None:
        """
        Initializes the EnvSpaces.

        Args:
            spaces: A dictionary containing the search space definition.
            pms: An optional SimpleNamespace object containing additional parameters.
        """

        self.natural_dim_ = spaces["dim"]
        self.true_dim_    = self.natural_dim_
        self.xmin_        = spaces["xmin"]
        self.xmax_        = spaces["xmax"]

        # These attributes may not be defined
        # get() defaults to None if the attribute is not present
        self.x0_     = spaces.get("x0")
        self.vmin_   = spaces.get("vmin")
        self.vmax_   = spaces.get("vmax")
        self.levels_ = spaces.get("levels")

        self.separable_ = False
        if hasattr(pms, "separable"): self.separable = pms.separable

        if (self.separable_):
            self.true_dim_ = 1

    @property
    def dim(self) -> int:
        """
        Returns the dimensionality of the search space.
        """
        return self.true_dim_

    @property
    def natural_dim(self):
        """
        Returns the natural dimensionality of the search space.
        """
        return self.true_dim_

    @property
    def x0(self) -> ndarray:
        """
        Returns the initial point in the search space.
        """
        return self.x0_

    @property
    def xmin(self) -> ndarray:
        """
        Returns the lower bounds of the search space.
        """
        return self.xmin_

    @property
    def xmax(self) -> ndarray:
        """
        Returns the upper bounds of the search space.
        """
        return self.xmax_

    @property
    def vmin(self) -> float:
        """
        Returns the minimum velocity.
        """
        return self.vmin_

    @property
    def vmax(self) -> float:
        """
        Returns the maximum velocity.
        """
        return self.vmax_

    @property
    def levels(self) -> List[float]:
        """
        Returns the levels.
        """
        return self.levels_
