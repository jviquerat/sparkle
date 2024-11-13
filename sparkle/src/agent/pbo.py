# Generic imports
import math
import numpy as np
import torch
import torch.distributions as td

# Custom imports
from sparkle.src.network.mlp import mlp
from sparkle.src.agent.base  import base_agent

###############################################
### PBO
class pbo(base_agent):
    def __init__(self, path, dim, x0, xmin, xmax, pms):

        super().__init__(pms)

        self.name        = "PBO"
        self.base_path   = path
        self.dim         = dim
        self.xmin        = xmin
        self.xmax        = xmax

        self.sigma0      = torch.tensor(0.25*(np.min(xmax)-np.max(xmin)))
        if hasattr(pms, "sigma0"):   self.sigma0   = torch.tensor(pms.sigma0)
        self.x0          = torch.tensor(x0)
        self.n_points    = 4 + math.floor(3.0*math.log(self.dim))
        if hasattr(pms, "n_points"): self.n_points = pms.n_points

        self.n_steps_max = 20
        if hasattr(pms, "n_steps_max"): self.n_steps_max = pms.n_steps_max
        self.n_elite     = int(0.5*self.n_points)
        if hasattr(pms, "n_elite"):     self.n_elite     = pms.n_elite

        self.adv_clip    = True
        if hasattr(pms, "adv_clip"):    self.adv_clip    = np.array(pms.adv_clip)
        self.adv_decay   = 1.0 - math.exp(-0.35*self.dim)
        if hasattr(pms, "adv_decay"):   self.adv_decay   = np.array(pms.adv_decay)
        self.obs_dim     = self.dim
        if hasattr(pms, "obs_dim"):     self.obs_dim     = pms.obs_dim
        self.cov_dim     = math.floor(self.dim*(self.dim - 1)/2)

        self.net_sg = mlp(inp_dim = self.obs_dim,
                          out_dim = self.dim,
                          arch    = pms.sg.arch,
                          acts    = pms.sg.acts,
                          lr      = pms.sg.lr)

        self.net_cr = mlp(inp_dim = self.obs_dim,
                          out_dim = self.cov_dim,
                          arch    = pms.cr.arch,
                          acts    = pms.cr.acts,
                          lr      = pms.cr.lr)

        self.sg_epochs = pms.sg.epochs
        self.sg_gen    = pms.sg.gen
        self.sg_batch  = pms.sg.batch

        self.cr_epochs = pms.cr.epochs
        self.cr_gen    = pms.cr.gen
        self.cr_batch  = pms.cr.batch

        self.obs = 0.0

        self.n_steps_total = self.n_steps_max*self.n_points
        self.n_steps_elite = self.n_steps_max*self.n_elite

        self.summary()

    # Reset
    def reset(self, run):

        # Mother class reset
        super().reset(run)

        # Additional data storage
        self.elite_stp    = 0
        self.hist_x_elite = np.zeros((self.n_steps_elite, self.dim))
        self.hist_a_elite = np.zeros((self.n_steps_elite))

        # Reset networks
        self.mu = torch.clone(self.x0)
        self.net_sg.reset()
        self.net_cr.reset()

        # Initial sampling
        # This fills x array with samples
        #return self.sample()

    # Sample from distribution
    def sample(self):

        obs = torch.ones(1,self.dim)*self.obs

        sg  = self.net_sg(obs)*self.sigma0
        cr  = self.net_cr(obs)

        pdf = self.get_pdf(self.mu, sg[0], cr[0])
        x   = pdf.sample([self.n_points]).numpy()

        return x

    # Compute full cov pdf
    def get_pdf(self, mu, sg, cr):

        cov = self.get_cov(sg, cr)

        #print(cov)
        scl = torch.linalg.cholesky(cov)
        pdf = td.MultivariateNormal(mu, scale_tril=scl)

        return pdf

    # Step
    def step(self, x, c):

        # Update best point
        self.update_best(x, c)

        # Store
        self.store(x, c)

        # Compute advantages
        # x is modified during this process
        x_elite = self.compute_advantages(x)

        # Update
        self.train_loop(self.sg_epochs, self.sg_gen,
                        self.sg_batch,  self.net_sg)
        self.train_loop(self.cr_epochs, self.cr_gen,
                        self.cr_batch,  self.net_cr)

        w = self.adv
        w = w/np.sum(w)
        self.mu[:] = 0.0
        for i in range(self.n_elite):
           self.mu[:] += w[i]*x_elite[i,:]

        self.stp += 1

    # Compute advantages
    def compute_advantages(self, x):

        # Update elite_step
        self.elite_stp += self.n_elite

        # Start and end indices of last generation
        # Here we retrieve all the sampled points of last
        # generation from the main score buffer
        start   = max(0,self.total_stp - self.n_points)
        end     = self.total_stp

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
        self.hist_a_elite[:] *= self.adv_decay
        self.hist_a_elite[start:end] = self.adv[:]

        return torch.tensor(x_elite)

    # Get data history
    def get_history(self, n_gen):

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

    # Train loop
    def train_loop(self, n_epochs, n_gens, batch_frac, net):

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
                net.opt_.zero_grad()
                loss.backward()
                net.opt_.step()

    # Compute loss
    def get_loss(self, obs, adv, act):

        # Compute pdf
        sg  = self.net_sg(obs)*self.sigma0
        cr  = self.net_cr(obs)

        pdf = self.get_pdf(self.mu, sg[0], cr[0])
        log = pdf.log_prob(act)

        # Compute loss
        s       = torch.multiply(adv, log)
        loss_pg =-torch.mean(s)

        return loss_pg

    # Compute covariance matrix
    def get_cov(self, sg, cr):

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
