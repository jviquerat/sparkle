import torch.optim as toptim


###############################################
class LBFGS():
    """
    LBFGS optimizer class.

    This class provides a wrapper around the PyTorch LBFGS optimizer,
    tailored for use with neural networks. It simplifies the process
    of creating and using the LBFGS optimizer, which is a quasi-Newton
    method for optimization.
    """
    def __init__(self, model, pms):
        """
        Initializes the LBFGS optimizer.

        Args:
            model: The model whose parameters will be optimized.
            pms: A SimpleNamespace object containing parameters for the optimizer,
                including history_size, max_iter, line_search, and lr.
        """

        self.history_size = 20
        self.max_iter     = 5
        self.line_search  = "strong_wolfe"
        self.lr           = 1.0

        if (hasattr(pms, "history_size")): self.history_size = pms.history_size
        if (hasattr(pms, "max_iter")):     self.max_iter     = pms.max_iter
        if (hasattr(pms, "line_search")):  self.line_search  = pms.line_search
        if (hasattr(pms, "lr")):           self.lr           = pms.lr

        self.model_ = model
        self.opt_   = toptim.LBFGS(model.params(),
                                   history_size=self.history_size,
                                   max_iter=self.max_iter,
                                   line_search_fn=self.line_search,
                                   lr=self.lr)

    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.

        This method calls the zero_grad() method of the underlying
        PyTorch LBFGS optimizer.
        """

        self.opt_.zero_grad()

    def step(self, closure):
        """
        Performs a single optimization step.

        This method calls the step() method of the underlying
        PyTorch LBFGS optimizer to update the model's parameters.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """

        self.opt_.step(closure)
