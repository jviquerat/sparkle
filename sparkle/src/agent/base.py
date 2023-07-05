# Generic imports
import random
import numpy as np

# Custom imports

###############################################
### Base agent
class base_agent():
    def __init__(self):
        pass

    # Get actions
    def actions(self, obs):
        raise NotImplementedError

    # Reset
    def reset(self):
        raise NotImplementedError

    # Save
    def save(self, filename):
        raise NotImplementedError

    # Load
    def load(self, filename):
        raise NotImplementedError
