# Generic imports
import torch
import torch.nn as tnn
from typing import Any, Iterator

torch.set_default_dtype(torch.double)

###############################################
### Base network class
class BaseNetwork(tnn.Module):
    def __init__(self) -> None:
        super().__init__()

    # Return network parameters
    def params(self) -> Iterator[Any]:

        return self.net_.parameters()

    # Dump network parameters to file
    def dump(self, filename: str) -> None:

        torch.save(self.net_.state_dict(), filename)

    # Load network parameters to file
    def load(self, filename: str) -> None:

        self.net_.load_state_dict(torch.load(filename))
