# Generic imports
import numpy as np

# Custom imports
from sparkle.src.utils.prints import *

###############################################
### Base experiment plan
class base_pex():
    def __init__(self, pms):
        pass

    # Return total nb of points
    def n_points(self):

        return self.n_points_

    # Return i-th point of pex
    def point(self, i):

        return np.array([self.x_[i]])

    # Return pex points
    def x(self):

        return self.x_

    # Print informations
    def summary(self):

        spacer()
        print("Pex type is "+self.name_+" with "+str(self.n_points())+" points")
