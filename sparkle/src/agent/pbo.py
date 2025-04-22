import math
import types
from typing import Tuple

import numpy as np
import torch
import torch.distributions as td
from numpy import ndarray
from torch.distributions.multivariate_normal import MultivariateNormal

from sparkle.src.agent.base import BaseAgent
from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.network.mlp import MLP
from sparkle.src.optimizer.adam import Adam
from sparkle.src.optimizer.optimizer import opt_factory
from sparkle.src.utils.default import set_default

###############################################

class PBO(BaseAgent):
    """
    Policy-Based Optimization (PBO) agent.

    This agent implements a Policy-Based Optimization algorithm, which uses
    neural networks to learn a policy for sampling new points in the search
    space. It adapts the sampling distribution based on the performance of
    the sampled points.
    """
    def __init__(self,
                 path: str,
                 spaces: EnvSpaces,
                 pms: types.SimpleNamespace) -> None:
        """
        Initializes the PBO agent.

        Args:
            path: The base path for storing results.
            spaces: The environment's search space definition.
            pms: A SimpleNamespace object containing parameters for the agent.
        """
        super().__init__(path, spaces, pms)

        self.name        = "PBO"
        sg0              = torch.tensor(0.25*(np.min(self.xmax)-np.max(self.xmin)))
        self.sigma0      = set_default("sigma0", sg0, pms)
        npts             = 4 + math.floor(3.0*math.log(self.dim))
        self.n_points    = set_default("n_points", npts, pms)
        self.n_steps_max = set_default("n_steps_max", 20, pms)
        self.n_elite     = set_default("n_elite", int(0.5*self.n_points), pms)
        self.adv_clip    = set_default("adv_clip", True, pms)
        self.adv_decay   = set_default("adv_decay", 1.0-math.exp(-0.35*self.dim), pms)
        self.obs_dim     = set_default("obs_dim", self.dim, pms)
        self.cov_dim     = math.floor(self.dim*(self.dim - 1)/2)

        # Create networks
        if (not hasattr(pms, "sg")): pms.sg = None
        self.sg_arch   = set_default("arch", [8,8], pms.sg)
        self.sg_acts   = set_default("acts", ["tanh", "tanh", "sigmoid"], pms.sg)
        self.sg_epochs = set_default("epochs", 8, pms.sg)
        self.sg_gen    = set_default("gen", 8, pms.sg)
        self.sg_batch  = set_default("batch", 0.5, pms.sg)
        self.net_sg = MLP(inp_dim = self.obs_dim,
                          out_dim = self.dim,
                          arch    = self.sg_arch,
                          acts    = self.sg_acts,
                          name    = "sigma")

        if (not hasattr(pms, "cr")): pms.cr = None
        self.cr_arch   = set_default("arch", [8,8], pms.cr)
        self.cr_acts   = set_default("acts", ["tanh", "tanh", "sigmoid"], pms.cr)
        self.cr_epochs = set_default("epochs", 8, pms.cr)
        self.cr_gen    = set_default("gen", 16, pms.cr)
        self.cr_batch  = set_default("batch", 1.0, pms.cr)
        self.net_cr = MLP(inp_dim = self.obs_dim,
                          out_dim = self.cov_dim,
                          arch    = self.cr_arch,
                          acts    = self.cr_acts,
                          name    = "correlations")

        # Create optimizers
        pms_opt_sg = types.SimpleNamespace()
        pms_opt_sg.type = "adam"
        pms_opt_sg.lr   = 5.0e-3
        self.pms_opt_sg = set_default("opt_sg", pms_opt_sg, pms)
        pms_opt_cr = types.SimpleNamespace()
        pms_opt_cr.type = "adam"
        pms_opt_cr.lr   = 1.0e-3
        self.pms_opt_cr = set_default("opt_cr", pms_opt_cr, pms)

        self.obs = 0.0

        self.n_steps_elite = self.n_steps_max*self.n_elite

        if (not self.silent): self.summary()
        if (not self.silent): self.net_sg.info()
        if (not self.silent): self.net_cr.info()

    def reset(self, run: int) -> None:
        """
        Resets the PBO agent for a new run.

        Args:
            run: The run number.
        """

        # Mother class reset
        super().reset(run)

        # Additional data storage
        self.elite_stp    = 0
        self.hist_c       = []
        self.hist_x_elite = np.zeros((self.n_steps_elite, self.dim))
        self.hist_a_elite = np.zeros((self.n_steps_elite))

        # Reset networks
        self.mu = torch.tensor(self.x0)
        self.net_sg.reset()
        self.net_cr.reset()

        # Create optimizers
        self.opt_sg = opt_factory.create(self.pms_opt_sg.type,
                                         model=self.net_sg,
                                         pms=self.pms_opt_sg)
        self.opt_cr = opt_factory.create(self.pms_opt_cr.type,
                                         model=self.net_cr,
                                         pms=self.pms_opt_cr)

    def sample(self) -> ndarray:
        """
        Samples new points from the PBO distribution.

        This method generates new points based on the current policy, which
        is represented by the neural networks that output the standard
        deviations and correlations.

        Returns:
            A NumPy array of shape (n_points, dim) representing the new points.
        """

        obs = torch.ones(1,self.dim)*self.obs

        sg  = self.net_sg(obs)*self.sigma0
        cr  = self.net_cr(obs)

        pdf = self.get_pdf(self.mu, sg[0], cr[0])
        x   = pdf.sample([self.n_points]).numpy()

        return x

    def get_pdf(self,
                mu: torch.Tensor,
                sg: torch.Tensor,
                cr: torch.Tensor) -> MultivariateNormal:
        """
        Computes the multivariate normal distribution.

        Args:
            mu: The mean vector.
            sg: The standard deviations.
            cr: The correlations.

        Returns:
            A MultivariateNormal distribution.
        """

        cov = self.get_cov(sg, cr)
        scl = torch.linalg.cholesky(cov)
        pdf = td.MultivariateNormal(mu, scale_tril=scl)

        return pdf

    def step(self, x: ndarray, c: ndarray) -> None:
        """
        Performs one step of the PBO algorithm.

        This method updates the policy networks based on the performance of
        the sampled points.

        Args:
            x: The points that were evaluated.
            c: The cost values at the evaluated points.
        """

        # Store costs
        for i in range(c.shape[0]):
            self.hist_c.append(c[i])

        # Compute advantages
        # x is modified during this process
        x_elite = self.compute_advantages(x)

        # Update
        self.train_loop(self.sg_epochs, self.sg_gen,
                        self.sg_batch,  self.net_sg, self.opt_sg)
        self.train_loop(self.cr_epochs, self.cr_gen,
                        self.cr_batch,  self.net_cr, self.opt_cr)

        w = self.adv
        w = w/np.sum(w)
        self.mu[:] = 0.0
        for i in range(self.n_elite):
           self.mu[:] += w[i]*x_elite[i,:]

        self.stp += 1

    def compute_advantages(self, x: ndarray) -> torch.Tensor:
        """
        Computes the advantages of the sampled points.

        Args:
            x: The points that were evaluated.

        Returns:
            A tensor of the elite points.
        """

        # Update elite_step
        self.elite_stp += self.n_elite

        # Start and end indices of last generation
        # Here we retrieve all the sampled points of last
        # generation from the main score buffer
        #start   = max(0,self.total_stp - self.n_points)
        #end     = self.total_stp
        start   = max(0,self.n_points*self.stp)
        end     = self.n_points*(self.stp+1)

        # Compute normalized advantage
        avg_rwd   = np.mean(self.hist_c[start:end])
        std_rwd   = np.std( self.hist_c[start:end])
        self.adv  = (self.hist_c[start:end] - avg_rwd)/(std_rwd + 1.0e-15)
        self.adv *=-1.0

        # Retain only elite points
        sc       = np.argsort(self.adv)
        x_elite  = x[sc[-self.n_elite:]].copy()
        self.adv = self.adv[sc[-self.n_elite:]]

        # Decay advantage history
        start = self.elite_stp - self.n_elite
        end   = self.elite_stp
        self.hist_x_elite[start:end] = x_elite[:]
        self.hist_a_elite[:]        *= self.adv_decay
        self.hist_a_elite[start:end] = self.adv[:]

        return torch.tensor(x_elite)

    def get_history(self, n_gen: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the history of elite points and their advantages.

        Args:
            n_gen: The number of generations to retrieve.

        Returns:
            A tuple containing:
                - A tensor of elite points.
                - A tensor of their advantages.
        """

        # Starting and ending indices based on the required nb of generations
        start     = max(0,self.elite_stp - n_gen*self.n_elite)
        end       = self.elite_stp
        buff_size = end - start

        # Randomize batch
        sample = np.arange(start, end)
        np.random.shuffle(sample)

        # Draw elements
        buff_x = torch.from_numpy(self.hist_x_elite[sample])
        buff_a = torch.from_numpy(self.hist_a_elite[sample])
        buff_x = torch.reshape(buff_x, [-1, self.dim])
        buff_a = torch.reshape(buff_a, [-1])

        return buff_x, buff_a

    def train_loop(self,
                   n_epochs: int,
                   n_gens: int,
                   batch_frac: float,
                   net: MLP,
                   opt: Adam) -> None:
        """
        Performs the training loop for the policy networks.

        Args:
            n_epochs: The number of training epochs.
            n_gens: The number of generations to use for training.
            batch_frac: The fraction of the history to use in each batch.
            net: The neural network to train.
            opt: The optimizer to use for training.
        """

        # Loop on epochs
        for epoch in range(n_epochs):
            x, a      = self.get_history(n_gens)
            n_samples = len(a)
            if (n_samples < 2*self.n_points): return
            done      = False
            btc       = 0
            n_batch   = max(math.ceil(n_samples*batch_frac), self.n_elite)

            # Visit all available history
            while not done:

                start    = btc*n_batch
                end      = min((btc+1)*n_batch,len(a))
                btc     += 1
                if (end == n_samples): done = True

                act = x[start:end]
                adv = a[start:end]
                obs = torch.ones((end-start, self.obs_dim))*self.obs

                loss = self.get_loss(obs, adv, act)
                opt.zero_grad()
                loss.backward()
                opt.step()

    def get_loss(self,
                 obs: torch.Tensor,
                 adv: torch.Tensor,
                 act: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss for the policy networks.

        Args:
            obs: The observations.
            adv: The advantages.
            act: The actions.

        Returns:
            The computed loss.
        """

        # Compute pdf
        sg  = self.net_sg(obs)*self.sigma0
        cr  = self.net_cr(obs)

        pdf = self.get_pdf(self.mu, sg[0], cr[0])
        log = pdf.log_prob(act)

        # Compute loss
        s       = torch.multiply(adv, log)
        loss_pg =-torch.mean(s)

        return loss_pg

    def get_cov(self,
                sg: torch.Tensor,
                cr: torch.Tensor) -> torch.Tensor:
        """
        Computes the covariance matrix.

        Args:
            sg: The standard deviations.
            cr: The correlations.

        Returns:
            The computed covariance matrix.
        """

        # Extract sigmas and thetas
        sigmas = sg
        thetas = cr*math.pi

        # Build initial theta matrix
        t = torch.ones([self.dim, self.dim])*math.pi/2.0
        t = torch.diagonal_scatter(t, torch.zeros(self.dim), offset=0)

        idx = 0
        for d in range(self.dim-1):
            diag = thetas[idx:idx+self.dim-(d+1)]
            idx += self.dim - (d+1)
            t    = torch.diagonal_scatter(t, diag, offset=-(d+1))
        cor = torch.cos(t)

        # Correct upper part to exact zero
        for d in range(self.dim-1):
            size = self.dim - (d+1)
            cor  = torch.diagonal_scatter(cor, torch.zeros(size), offset=(d+1))

        # Roll and compute additional terms
        for roll in range(self.dim-1):
            vec = torch.ones([self.dim, 1])
            vec = torch.mul(vec, math.pi/2.0)
            t   = torch.cat([vec, t[:, :self.dim-1]], axis=1)

            for d in range(self.dim-1):
                zero = torch.zeros(self.dim - (d+1))
                t    = torch.diagonal_scatter(t, zero, offset=(d+1))

            cor = torch.mul(cor, torch.sin(t))

        cor = torch.matmul(cor, torch.transpose(cor,0,1))
        scl = torch.zeros([self.dim, self.dim])
        scl = torch.diagonal_scatter(scl, torch.sqrt(sigmas), offset=0)
        cov = torch.matmul(scl, cor)
        cov = torch.matmul(cov, scl)

        return cov
