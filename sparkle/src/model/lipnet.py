import os
import math
from typing import List, Tuple
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as toptim
from numpy import ndarray
from torch.autograd import grad as autograd
torch.set_default_dtype(torch.double)

from sparkle.src.kernel.gaussian import Gaussian
from sparkle.src.model.base import BaseModel
from sparkle.src.network.lip_mlp import LipMLP
from sparkle.src.utils.default import set_default
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
        self.ensemble_size = set_default("ensemble_size", 5, pms)
        self.n_epochs_max  = set_default("n_epochs_max", 50000, pms)
        self.target_loss   = set_default("target_loss", 1.0e-4, pms)
        self.lr            = set_default("lr", 5.0e-4, pms)
        self.beta          = set_default("beta", 0.0, pms)
        self.load_model_   = set_default("load_model", False, pms)
        self.arch          = set_default("arch", [16*self.spaces.dim]*2, pms)
        self.acts          = set_default("acts", ["tanh", "tanh", "linear"], pms)
        self.lip_constants = set_default("lip_constant", [2.0, 2.0, 2.0], pms)

        self.ensemble: List[LipMLP] = []
        for i in range(self.ensemble_size):
            member = LipMLP(
                inp_dim=self.spaces.dim,
                out_dim=1,
                arch=self.arch,
                acts=self.acts,
                lip_constant=self.lip_constants,
                name=f"lipmlp_member_{i}"
            )
            self.ensemble.append(member)

    def reset(self):
        """
        Resets the model to its initial state.
        """
        for model in self.ensemble:
            model.reset()

    def build(self, x: ndarray, y: ndarray):
        """
        Builds and trains the model from input data.

        Args:
            x: Training input data.
            y: Training output data.
        """
        self.ymax_ = np.max(y)
        self.ymin_ = np.min(y)
        self.x_    = self.normalize(x, self.spaces.xmin, self.spaces.xmax)
        self.y_    = self.normalize(y, self.ymin_, self.ymax_)

        n_points   = x.shape[0]
        batch_size = math.floor(1.0*n_points)
        all_loss   = []

        for i, model in enumerate(self.ensemble):
            optimizer = toptim.Adam(model.params(), lr=self.lr)

            epoch        = 0
            loss_val     = 1.0e8
            history_loss = []
            while loss_val > self.target_loss and epoch < self.n_epochs_max:
                epoch_loss = 0.0
                p = np.random.permutation(n_points)
                xb_np, cb_np = self.x_[p], self.y_[p]

                done = False
                btc = 0
                while not done:
                    start, end = btc*batch_size, min((btc + 1)*batch_size, n_points)
                    if start >= end: break
                    btc += 1
                    if end == n_points: done = True

                    xb = torch.from_numpy(xb_np[start:end])
                    cb = torch.from_numpy(cb_np[start:end])

                    loss = self.get_loss(model, xb, cb)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_epoch_loss = epoch_loss / float(btc) if btc > 0 else 0.0
                history_loss.append([epoch, avg_epoch_loss])
                loss_val = avg_epoch_loss

                spacer(f"Model #{i}, epoch #{epoch}, loss = {fmt_float(avg_epoch_loss)}                  ", end="\r")
                epoch += 1

            all_loss.append(history_loss)

        self.plot_loss(self.path+"/loss.png", all_loss)

        new_line()

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
        lip = np.zeros(n_batch)

        xx = self.normalize(x, self.spaces.xmin, self.spaces.xmax)
        xx = torch.from_numpy(xx)

        y_all = np.zeros((self.ensemble_size, n_batch))
        with torch.no_grad():
            for i, model in enumerate(self.ensemble):
                y = model.forward(xx)
                y_all[i,:] = y.squeeze().numpy()

        y_mean_norm = np.mean(y_all, axis=0)
        y_std_norm = np.std(y_all, axis=0)

        y_mean = self.denormalize(y_mean_norm, self.ymin_, self.ymax_)
        y_std = y_std_norm*(self.ymax_ - self.ymin_)

        return y_mean, y_std

    def evaluate_grad(self, x: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Evaluates the gradient of the ensemble mean prediction AND
        the gradient of the ensemble standard deviation prediction
        with respect to the input.

        Args:
            x: Input data of shape (n_batch, input_dim).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (grad_mean, grad_std),
                representing nabla[mu(x)] and nabla[sigma(x)],
                properly scaled to original input/output space.
        """
        n_batch = x.shape[0]
        dim     = x.shape[1]
        xx      = self.normalize(x, self.spaces.xmin, self.spaces.xmax)
        xx      = torch.from_numpy(xx).requires_grad_(True)

        # Store individual model predictions
        y_list = []
        for model in self.ensemble:
            y_list.append(model.forward(xx))

        # Stack predictions: shape (ensemble_size, n_batch, 1) or (ensemble_size, n_batch)
        y_all = torch.stack(y_list, dim=0)
        if y_all.dim() == 3 and y_all.shape[-1] == 1:
            y_all = y_all.squeeze(-1) # shape (ensemble_size, n_batch)

        # Compute mean and std
        y_mean = torch.mean(y_all, dim=0) # shape (n_batch,)
        y_std = torch.std(y_all, dim=0) # shape (n_batch,)
        y_std = y_std + 1e-5 # prevent 0 std

        # Gradients of mean
        grad_outputs_mean = torch.ones_like(y_mean)
        grad_of_mean = autograd(
            outputs=y_mean,
            inputs=xx,
            grad_outputs=grad_outputs_mean,
            retain_graph=True, # Keep graph for next autograd call
            create_graph=False
        )[0]
        grad_mean = grad_of_mean.detach().clone().numpy()

        # Gradients of std
        grad_outputs_std = torch.ones_like(y_std)
        grad_of_std = autograd(
            outputs=y_std,
            inputs=xx,
            grad_outputs=grad_outputs_std,
            retain_graph=False,
            create_graph=False
        )[0]
        grad_std = grad_of_std.detach().numpy()

        # Denormalize
        grad_scale  = (self.ymax_-self.ymin_)/(self.spaces.xmax-self.spaces.xmin)
        grad_mean  *= grad_scale
        grad_std   *= grad_scale

        return grad_mean, grad_std

    def get_loss(self, model: LipMLP, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss.

        Args:
            model: current lipschitz model
            x: Input data of shape (batch_size, dim).
            c: Target values of shape (batch_size,).

        Returns:
            torch.Tensor: The computed loss.
        """
        r = model.forward(x.requires_grad_(True))
        mse_loss = torch.mean((c - r.reshape(-1)) ** 2)

        lip_constants = model.lip_consts()
        lip_penalty = sum(abs(lc) for lc in lip_constants)

        total_loss = mse_loss + self.beta*lip_penalty

        return total_loss

    def plot_loss(self, filename: str, loss: List[List[float]]):
        """
        Plots the training loss curve.

        Args:
            filename: The name of the file to save the plot to.
            loss: Loss data of shape (n_samples, n_losses+1).
            labels: Labels for each loss curve.
        """
        plt.clf()
        fig = plt.figure()
        for i in range(len(loss)):
            arr = np.array(loss[i])
            plt.plot(arr[:,0], arr[:,1])
        plt.yscale("log")

        plt.savefig(filename, dpi=100)
        plt.close()

    def dump(self, filename_prefix="lipnet_member_"):
        pass
