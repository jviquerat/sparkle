# Generic imports
import numpy as np

###############################################
### A class holding informations dimensions
class environment_spaces:
    def __init__(self, spaces, pms=None):

        self.natural_dim_ = spaces[0]
        self.true_dim_    = self.natural_dim_
        self.x0_          = spaces[1]
        self.xmin_        = spaces[2]
        self.xmax_        = spaces[3]

        if (len(spaces) > 4):
            self.vmin_   = spaces[4]
            self.vmax_   = spaces[5]
            self.levels_ = spaces[6]

        self.separable_ = False
        if hasattr(pms, "separable"): self.separable = pms.separable

        if (self.separable_):
            self.true_dim_ = 1

    # Accessor
    def dim(self):
        return self.true_dim_

    # Accessor
    def natural_dim(self):
        return self.true_dim_

    # Accessor
    def x0(self):
        return self.x0_

    # Accessor
    def xmin(self):
        return self.xmin_

    # Accessor
    def xmax(self):
        return self.xmax_

    # Accessor
    def vmin(self):
        return self.vmin_

    # Accessor
    def vmax(self):
        return self.vmax_

    # Accessor
    def levels(self):
        return self.levels_
