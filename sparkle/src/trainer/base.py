# Generic imports
import os
import math
import shutil
import numpy as np

# Custom imports

###############################################
### Base trainer
class base_trainer():
    def __init__(self):
        pass

    # Optimize
    def optimize(self):
        raise NotImplementedError

    # Reset
    def reset(self):
        raise NotImplementedError

    # Printings
    def print(self):
        raise NotImplementedError
