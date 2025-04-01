# Generic imports
import torch
import torch.nn as tnn
torch.set_default_dtype(torch.double)

###############################################
### Base network class
class BaseNetwork(tnn.Module):
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
