# Generic imports
import math
import numpy as np
import torch
import torch.nn as tnn
import torch.optim as toptim

# Custom imports
from sparkle.src.network.torch_dicts import *
from sparkle.src.utils.prints import *

###############################################
### Base network class
class base(tnn.Module):
    def __init__(self):
        super().__init__()

    # Return network parameters
    def params(self):

        return self.net_.parameters()

    # Dump network parameters to file
    def dump(self, filename):

        torch.save(self.net_.state_dict(), filename)

    # Load network parameters to file
    def load(self, filename):

        self.net_.load_state_dict(torch.load(filename))
