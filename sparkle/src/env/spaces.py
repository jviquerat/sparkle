from types import SimpleNamespace
from typing import Any, List, Optional

from numpy import ndarray


###############################################
### A class holding informations dimensions
class EnvSpaces:
    def __init__(self,
                 spaces: Any,
                 pms: Optional[SimpleNamespace]=None) -> None:

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
        return self.true_dim_

    @property
    def natural_dim(self):
        return self.true_dim_

    @property
    def x0(self) -> ndarray:
        return self.x0_

    @property
    def xmin(self) -> ndarray:
        return self.xmin_

    @property
    def xmax(self) -> ndarray:
        return self.xmax_

    @property
    def vmin(self) -> float:
        return self.vmin_

    @property
    def vmax(self) -> float:
        return self.vmax_

    @property
    def levels(self) -> List[float]:
        return self.levels_
