from types import SimpleNamespace

import torch.optim as toptim

from sparkle.src.network.mlp import MLP


###############################################
class AdamW():
    """
    AdamW optimizer class.

    This class provides a wrapper around the PyTorch AdamW optimizer,
    tailored for use with the MLP (Multi-Layer Perceptron) network.
    It simplifies the process of creating and using the AdamW optimizer
    for training neural networks, which is a variant of Adam that
    incorporates weight decay.
    """
    def __init__(self, model: MLP, pms: SimpleNamespace) -> None:
        """
        Initializes the AdamW optimizer.

        Args:
            model: The MLP model whose parameters will be optimized.
            pms: A SimpleNamespace object containing parameters for the optimizer,
                including the learning rate (lr).
        """

        self.model_ = model
        self.opt_   = toptim.AdamW(self.model_.params(),
                                  lr=pms.lr)

    def zero_grad(self) -> None:
        """
        Clears the gradients of all optimized parameters.

        This method calls the zero_grad() method of the underlying
        PyTorch AdamW optimizer.
        """

        self.opt_.zero_grad()

    def step(self) -> None:
        """
        Performs a single optimization step.

        This method calls the step() method of the underlying
        PyTorch AdamW optimizer to update the model's parameters.
        """

        self.opt_.step()
