import os
import math
from typing import List, Tuple
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as toptim
import torch.optim.lr_scheduler as tsched
from numpy import ndarray
from torch.autograd import grad as autograd
torch.set_default_dtype(torch.double)

from sparkle.src.kernel.gaussian import Gaussian
from sparkle.src.model.base import BaseModel
from sparkle.src.network.lip_mlp import LipMLP
from sparkle.src.utils.default import set_default
from sparkle.src.utils.prints import spacer, fmt_float, new_line


class LipNet(BaseModel):
    """
    A Lipschitz-constrained neural network with fast geometric ensembling
    FGE aims at generating snapshots from different local minima while avoiding
    the entire retraining of different models.
    After a first "regular" training step, a sawtooth-like cyclic learning rate
    is applied to collect snapshots of retrained networks and use them for ensembling
    """
    def __init__(self, spaces, path, pms):
        """
        Initializes the model

        Args:
            spaces: Search space definition.
            path: Path to store artifacts.
            pms: Model parameters.
        """
        super().__init__(spaces, path)

        self.fge_cycles = set_default("fge_cycles", 5, pms)
        self.fge_first_cycle_len = set_default("fge_first_cycle_len", 15000, pms)
        self.fge_cycle_len = set_default("fge_cycle_len", 2000, pms)

        self.n_epochs_max = self.fge_first_cycle_len + (self.fge_cycles-1)*self.fge_cycle_len

        self.target_loss = set_default("target_loss", 1.0e-4, pms)
        self.lr = set_default("lr", 5.0e-4, pms)
        self.beta = set_default("beta", 0.0, pms)
        self.load_model_ = set_default("load_model", False, pms)

        width = 16*math.floor(math.sqrt(self.spaces.dim))
        self.arch = set_default("arch", [width]*2, pms)
        self.acts = set_default("acts", ["tanh", "tanh", "linear"], pms)
        self.lip_constants = set_default("lip_constants", [2.0, 2.0, 2.0], pms)

        self.noise_level = set_default("noise_level", 0.05, pms)
        self.patience = set_default("patience", 1000, pms)
        self.improvement = set_default("improvement", 1.0e-6, pms)

        self.model: LipMLP = LipMLP(
            inp_dim=self.spaces.dim,
            out_dim=1,
            arch=self.arch,
            acts=self.acts,
            lip_constants=self.lip_constants,
            name="lipmlp_fge"
        )

    def reset(self):
        """
        Resets the model to its initial state.
        """
        self.model.reset()

    def build(self, x: ndarray, y: ndarray):
        """
        Builds and trains the FGE model from input data.

        Args:
            x: Training input data.
            y: Training output data.
        """
        self.ymax_ = np.max(y)
        self.ymin_ = np.min(y)
        self.x_ = self.normalize(x, self.spaces.xmin, self.spaces.xmax)
        self.y_ = self.normalize(y, self.ymin_, self.ymax_)

        self.fge_snapshots_weights = []

        n_points = x.shape[0]
        batch_size = math.floor(1.0*n_points)
        loss_history = []

        total_epochs = 0
        for cycle in range(self.fge_cycles):
            optimizer = toptim.Adam(self.model.params(), lr=self.lr)
            if cycle == 0:
                cycle_epochs = self.fge_first_cycle_len
                scheduler = tsched.CosineAnnealingLR(optimizer,
                                                     T_max=self.fge_first_cycle_len)
            else:
                # Reset weigths to first run
                self.set_snapshot_weights(0)
                
                cycle_epochs = self.fge_cycle_len
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.lr
                scheduler = tsched.CosineAnnealingLR(optimizer,
                                                     T_max=self.fge_cycle_len)

                with torch.no_grad():
                    for param in self.model.parameters():
                        param.add_(self.noise_level*(1.0 - 2.0*torch.rand_like(param)))

            previous_loss = float('inf')
            n_wait = 0
            for epoch in range(cycle_epochs):
                total_epochs += 1
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

                    loss = self.get_loss(self.model, xb, cb)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_epoch_loss = epoch_loss / float(btc) if btc > 0 else 0.0
                loss_history.append([total_epochs, avg_epoch_loss])
                scheduler.step()

                if (abs(loss - previous_loss)/loss) < self.improvement:
                    n_wait +=1
                else:
                    n_wait = 0

                if n_wait == self.patience:
                    break

                previous_loss = loss

                cycle_text = f"FGE Cycle {cycle}"
                epoch_text = f"epoch {epoch}"
                lr = f"{fmt_float(optimizer.param_groups[0]['lr'])}"
                loss = f"{fmt_float(avg_epoch_loss)}"
                spacer(f"{cycle_text}, {epoch_text}, loss = {loss}, lr = {lr}                     ", end="\r")

            # Save snapshot
            snapshot_weights = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
            self.fge_snapshots_weights.append(snapshot_weights)

        self.plot_loss(self.path+"/loss.png", loss_history)
        new_line()

        # Set model to average weights
        self.set_avg_weights()

    def set_avg_weights(self):
        """
        Sets the model weights to the average of FGE snapshots
        """
        avg_weights = {}
        num_snapshots = len(self.fge_snapshots_weights)
        for key in self.fge_snapshots_weights[0].keys():
            avg_weights[key] = sum(snapshot[key] for snapshot in self.fge_snapshots_weights)/num_snapshots

        self.model.load_state_dict(avg_weights)

    def set_snapshot_weights(self, k):
        """
        Sets the model weights to a specific snapshot
        """
        weights = {}
        for key in self.fge_snapshots_weights[0].keys():
            weights[key] = self.fge_snapshots_weights[k][key]

        self.model.load_state_dict(weights)

    def evaluate(self, x: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Evaluates the FGE model at test points.
        The mean is from the averaged snapshots model (or individual snapshots).
        The std is from the spread of snapshot predictions.

        Args:
            x: Input data of shape (n_batch, input_dim).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (y_mean, y_std), where y_mean is the
                                           predicted output and y_std is the uncertainty.
        """
        n_batch = x.shape[0]
        xx_np = self.normalize(x, self.spaces.xmin, self.spaces.xmax)
        xx = torch.from_numpy(xx_np)

        y_mean_norm_np = np.zeros(n_batch)
        y_std_norm_np = np.zeros(n_batch)
        y_all_snapshots = np.zeros((len(self.fge_snapshots_weights), n_batch))

        # Load successive snapshots and infer output
        for i, snapshot_weights in enumerate(self.fge_snapshots_weights):
            weights = {k: v for k,v in snapshot_weights.items()}
            self.model.load_state_dict(weights)
            with torch.no_grad():
                y_all_snapshots[i,:] = self.model.forward(xx).squeeze().numpy()

        # Restore averaged model weights
        self.set_avg_weights()

        # Compute mean and std of outputs
        y_mean_norm_np = np.mean(y_all_snapshots, axis=0)
        y_std_norm_np = np.std(y_all_snapshots, axis=0)

        y_mean = self.denormalize(y_mean_norm_np, self.ymin_, self.ymax_)
        y_std = y_std_norm_np*(self.ymax_ - self.ymin_)

        return y_mean, y_std


    def evaluate_grad(self, x: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Evaluates the gradient of the FGE mean prediction AND
        the gradient of the FGE standard deviation prediction
        with respect to the input.

        Args:
            x: Input data of shape (n_batch, input_dim).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (grad_mean, grad_std),
                                           representing nabla[mu(x)] and nabla[sigma(x)],
                                           properly scaled to original input/output space.
        """
        n_batch = x.shape[0]
        xx = self.normalize(x, self.spaces.xmin, self.spaces.xmax)
        xx = torch.from_numpy(xx).requires_grad_(True)

        grad_mean = np.zeros((n_batch, self.spaces.dim))
        grad_std = np.zeros((n_batch, self.spaces.dim))

        # Gradient of mean
        self.set_avg_weights()
        y_mean = self.model.forward(xx).squeeze()

        grad_outputs_mean = torch.ones_like(y_mean)
        grad_of_mean = autograd(
            outputs=y_mean,
            inputs=xx,
            grad_outputs=grad_outputs_mean,
            retain_graph=True, # Keep graph for next autograd call
            create_graph=False
        )[0]
        grad_mean = grad_of_mean.detach().clone().numpy()

        # Gradient of std
        y_list_snapshots = []
        for snapshot_weights in self.fge_snapshots_weights:
            weights = {k: v for k,v in snapshot_weights.items()}
            self.model.load_state_dict(weights)
            y_list_snapshots.append(self.model.forward(xx).squeeze())

        y_all_snapshots = torch.stack(y_list_snapshots, dim=0)
        y_std_fge = torch.std(y_all_snaps, dim=0) + 1.0e-5

        grad_outputs_std = torch.ones_like(y_std_fge)
        grad_of_std = autograd(
            outputs=y_std_fge,
            inputs=xx,
            grad_outputs=grad_outputs_std,
            retain_graph=False,
            create_graph=False
            )[0]
        grad_std = grad_of_std.detach().numpy()

        # Denormalize
        grad_scale = (self.ymax_-self.ymin_)/(self.spaces.xmax-self.spaces.xmin)
        grad_mean *= grad_scale
        grad_std  *= grad_scale

        # Restore averaged model weights
        self.set_avg_weights()

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
        r = model.forward(x.requires_grad_(True)) # As per original user code
        mse_loss = torch.mean((c.squeeze() - r.squeeze()) ** 2)

        lip_constants = model.lip_consts()
        lip_penalty = sum(abs(lc) for lc in lip_constants)
        total_loss = mse_loss + self.beta*lip_penalty

        return total_loss

    def plot_loss(self, filename: str, loss_data: List[List[float]]):
        """
        Plots the training loss curve.

        Args:
            filename: The name of the file to save the plot to.
            loss_data: A list containing one list of [epoch, loss_value] pairs.
        """
        plt.clf()
        fig = plt.figure()

        arr = np.array(loss_data)
        plt.plot(arr[:,0], arr[:,1])
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("FGE Training Loss")
        plt.savefig(filename, dpi=100)
        plt.close(fig)

    def dump(self, filename_prefix="lipnet_fge_model"):
        """
        Model saving is disabled as per request.
        """
        pass
