from typing import Any, Iterator

import torch
import torch.nn as tnn

torch.set_default_dtype(torch.double)


class BaseNetwork(tnn.Module):
    """
    Base class for neural networks.

    This class defines the common interface and functionality for all neural
    networks used in the optimization framework. It provides methods for
    managing network parameters, saving, and loading network states.
    """
    def __init__(self) -> None:
        """
        Initializes the BaseNetwork.
        """
        super().__init__()

    def params(self) -> Iterator[Any]:
        """
        Returns an iterator over the network parameters.

        Returns:
            An iterator over the network parameters.
        """
        return self.net_.parameters()

    def dump(self, filename: str) -> None:
        """
        Dumps the network parameters to a file.

        Args:
            filename: The name of the file to which to dump the parameters.
        """
        torch.save(self.net_.state_dict(), filename)

    def load(self, filename: str) -> None:
        """
        Loads the network parameters from a file.

        Args:
            filename: The name of the file from which to load the parameters.
        """
        self.net_.load_state_dict(torch.load(filename))
