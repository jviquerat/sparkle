# Generic imports
import random
import numpy as np

# Custom imports

###############################################
### Base agent
class base_agent():
    def __init__(self):
        pass

    # Perform one optimization step
    def step(self):
        raise NotImplementedError

    # Return degrees of freedom
    def dof(self):
        raise NotImplementedError

    # Reset
    def reset(self):
        raise NotImplementedError

    # Render
    def render(self):
        raise NotImplementedError
