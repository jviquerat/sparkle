# Generic imports
import numpy as np

###############################################
### A class holding informations dimensions
class environment_spaces:
    def __init__(self, spaces, pms=None):

        self.dim_  = spaces[0]
        self.x0_   = spaces[1]
        self.xmin_ = spaces[2]
        self.xmax_ = spaces[3]

    # Accessor
    def dim(self):
        return self.dim_

    # Accessor
    def x0(self):
        return self.x0_

    # Accessor
    def xmin(self):
        return self.xmin_

    # Accessor
    def xmax(self):
        return self.xmax_
