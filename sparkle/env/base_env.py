# Generic imports
import math
import numpy as np

###############################################
### Base environment
class base_env():

    ### Create object
    def __init__(self):
        pass

    ### Cost function
    def cost(self, x):
        raise NotImplementedError

    ### Rendering
    def render(self):
        pass
