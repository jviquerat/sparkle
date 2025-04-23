import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as toptim
from numpy import ndarray
from torch.autograd import grad as autograd

from sparkle.src.kernel.gaussian import Gaussian
from sparkle.src.model.base import BaseModel
from sparkle.src.network.lip_mlp import LipMLP
from sparkle.src.utils.prints import spacer, fmt_float, new_line


class LipNet(BaseModel):
    def __init__(self, spaces, path, pms):
        """
        Initializes the Lipschitzian network model.

        Args:
            spaces: Search space definition.
            path: Path to store artifacts.
            pms: Model parameters.
        """
        super().__init__(spaces, path)
        self.n_epochs_max = pms.n_epochs_max
        self.target_loss = pms.target_loss
        self.lr = pms.lr
        self.beta = pms.beta
        self.reset()

    def reset(self):
        """
        Resets the model to its initial state.
        """
        size = 16*self.spaces.dim
        self.net = LipMLP(inp_dim=self.spaces.dim,
                          out_dim=1,
                          arch=[size, size],
                          acts=["tanh", "tanh", "linear"],
                          lip_constant=[2.0, 2.0, 2.0])

        self.opt = toptim.Adam(self.net.params(),
                               lr=self.lr)
        self.kernel = Gaussian(self.spaces)

    def evaluate(self, x: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Evaluates the model at test points.

        Args:
            x: Input data of shape (n_batch, input_dim).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (y, acq), where y is the predicted
                output and acq is the acquisition function value.
        """
        n_batch = x.shape[0]
        y_out = np.zeros(n_batch)
        lip = np.zeros(n_batch)

        xx = self.normalize(x, self.spaces.xmin, self.spaces.xmax)

        for k in range(n_batch):
            xt = torch.tensor(xx[k]).requires_grad_(True)
            y = self.net.forward(xt)

            grad_outputs = torch.ones_like(y)
            grad = autograd(outputs=y,
                            inputs=xt,
                            create_graph=True,
                            grad_outputs=grad_outputs)[0]
            local_lip = grad.norm()

            lip[k] = local_lip.detach().item()
            y_out[k] = y.detach().detach().item()

        density = np.sum(self.kernel.covariance(xx, self.x_), axis=1)
        density /= np.max(density)

        # Denormalize evaluation
        y  = self.denormalize(y_out, self.ymin_, self.ymax_)
        #imp = np.maximum(self.ymin_ - y, 0.0)
        imp = self.sigmoid(self.ymin_ - y)

        acq = imp*lip/density
        #acq = imp*sample_density
        #acq = np.log(acq)

        # density = np.mean(self.kernel.covariance(xx, self.x_), axis=1)
        # y = self.denormalize(y_out, self.ymin_, self.ymax_)
        # imp = np.maximum(self.ymin_ - y, 0.0)
        # acq = imp * lip * density

        return y, lip #imp*lip*density

    def sigmoid(self, x: ndarray) -> ndarray:
        """
        Applies the sigmoid function.

        Args:
            x: Input array.

        Returns:
            np.ndarray: Output array after applying sigmoid.
        """
        return 1.0 / (1.0 + np.exp(-x))

    def build(self, x: ndarray, y: ndarray):
        """
        Builds and trains the model from input data.

        Args:
            x: Training input data.
            y: Training output data.
        """
        self.reset()
        self.ymax_ = np.max(y)
        self.ymin_ = np.min(y)
        self.x_    = self.normalize(x, self.spaces.xmin, self.spaces.xmax)
        self.y_    = self.normalize(y, self.ymin_, self.ymax_)

        self.kernel.optimize(self.x_, self.y_)

        n_points     = x.shape[0]
        batch_size   = math.floor(1.0*n_points)
        history_loss = np.zeros((10*self.n_epochs_max, 2))
        loss = 1.0e8

        #for epoch in range(self.n_epochs):
        epoch = 0
        while loss > self.target_loss:
            r = np.arange(0, n_points)
            p = np.random.permutation(r)
            xb = torch.from_numpy(self.x_[p])
            c = torch.from_numpy(self.y_[p])

            done = False
            btc = 0

            while not done:
                start = btc * batch_size
                end = min((btc + 1) * batch_size, n_points)
                btc += 1
                if (end == n_points):
                    done = True

                loss = self.get_loss(xb[start:end], c[start:end])

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                history_loss[epoch, 1] += loss

            history_loss[epoch, 0] = epoch
            history_loss[epoch, 1] /= float(btc)
            spacer(f"Epoch #{epoch}, loss = {fmt_float(history_loss[epoch, 1])}                     ", end="\r")

            if (epoch == self.n_epochs_max-1): break
            epoch += 1

        new_line()
        self.plot_loss(self.path + "/loss.png", history_loss, ["loss"])
        spacer(f"Lipschitz constants: {self.net.lip_consts()}")

    def get_loss(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss.

        Args:
            x: Input data of shape (batch_size, dim).
            c: Target values of shape (batch_size,).

        Returns:
            torch.Tensor: The computed loss.
        """
        r = self.net.forward(x.requires_grad_(True))
        mse_loss = torch.mean((c - r.reshape(-1)) ** 2)

        lip_constants = self.net.lip_consts()
        lip_penalty = sum(abs(lc) for lc in lip_constants)

        total_loss = mse_loss + self.beta*lip_penalty

        return total_loss

    # Dump model
    def dump(self, filename="lipnet.dat"):
        """
        Placeholder for dumping the model to a file.

        Args:
            filename: The name of the file to save
                the model to. Defaults to "lipnet.dat".
        """

    def plot_loss(self, filename: str, loss: ndarray, labels: List[str]):
        """
        Plots the training loss curve.

        Args:
            filename: The name of the file to save the plot to.
            loss: Loss data of shape (n_samples, n_losses+1).
            labels: Labels for each loss curve.
        """
        plt.clf()
        fig = plt.figure()
        for k in range(loss.shape[1] - 1):
            plt.plot(loss[:, 0], loss[:, k + 1], label=labels[k])
        plt.yscale("log")
        plt.legend(loc="upper right")
        plt.savefig(filename, dpi=100)
        plt.close()
