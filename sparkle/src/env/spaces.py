# Generic imports
import numpy as np

###############################################
### A class holding informations dimensions
class env_spaces:
    def __init__(self, spaces, pms=None):

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
    def dim(self):
        return self.true_dim_

    @property
    def natural_dim(self):
        return self.true_dim_

    @property
    def x0(self):
        return self.x0_

    @property
    def xmin(self):
        return self.xmin_

    @property
    def xmax(self):
        return self.xmax_

    @property
    def vmin(self):
        return self.vmin_

    @property
    def vmax(self):
        return self.vmax_

    @property
    def levels(self):
        return self.levels_
