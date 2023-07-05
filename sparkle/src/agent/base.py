# Generic imports
import random
import numpy as np

# Custom imports

###############################################
### Base agent
class base_agent():
    def __init__(self):
        pass

    # Perform optimization
    def optimize(self):
        raise NotImplementedError

    # Reset
    def reset(self):
        raise NotImplementedError
