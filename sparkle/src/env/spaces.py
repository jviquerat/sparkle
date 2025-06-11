from types import SimpleNamespace
from typing import Any, List, Optional

from numpy import ndarray


###############################################
class EnvSpaces:
    """
    A class holding information about the search space.

    This class stores and manages information about the search space,
    including its dimensionality, bounds, and other relevant properties.

    Some methods may use mixt spaces (continuous/discrete)
    As this is not the default case, the base functions (dim, xmin, xmax...)
    are intended for continuous problems
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

        self.dim_ = spaces["dim"]
        self.xmin_        = spaces["xmin"]
        self.xmax_        = spaces["xmax"]

        # These attributes may not be defined
        # get() defaults to None if the attribute is not present
        self.x0_     = spaces.get("x0")
        self.vmin_   = spaces.get("vmin")
        self.vmax_   = spaces.get("vmax")
        self.levels_ = spaces.get("levels")

    @property
    def dim(self) -> int:
        """
        Returns the dimensionality of the search space.
        """
        return self.dim_

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
