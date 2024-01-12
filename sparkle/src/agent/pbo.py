# Custom imports
from sparkle.src.network.fc import *
from sparkle.src.agent.base import *

import torch.distributions as td

###############################################
### PBO
class pbo(base_agent):
    def __init__(self, path, dim, xmin, xmax, pms):

        super().__init__(pms)

        self.name        = "PBO"
        self.base_path   = path
        self.dim         = dim
        self.xmin        = xmin
        self.xmax        = xmax

        self.sigma0      = 0.25*(np.min(xmax)-np.max(xmin))
        self.x0          = 0.5*(xmax+xmin)
        self.n_points    = 4 + math.floor(3.0*math.log(self.dim))

        self.n_steps_max = 20
        self.p_elite     = 2
        self.n_elite     = self.n_points//self.p_elite

        self.adv_clip    = True
        self.adv_decay   = 1.0 - math.exp(-0.35*self.dim)
        self.obs_dim     = self.dim
        self.cov_dim     = math.floor(self.dim*(self.dim - 1)/2)

        if hasattr(pms, "n_steps_max"): self.n_steps_max = pms.n_steps_max
        if hasattr(pms, "n_points"):    self.n_points    = pms.n_points
        if hasattr(pms, "obs_dim"):     self.obs_dim     = pms.obs_dim
        if hasattr(pms, "sigma0"):      self.sigma0      = pms.sigma0
        if hasattr(pms, "x0"):          self.x0          = np.array(pms.x0)

        if hasattr(pms, "adv_clip"):    self.adv_clip    = np.array(pms.adv_clip)
        if hasattr(pms, "adv_decay"):   self.adv_decay   = np.array(pms.adv_decay)

        self.net_mu = fc(inp_dim=self.obs_dim,
                         arch=pms.mu.arch,
                         acts=pms.mu.acts,
                         lr=pms.mu.lr)

        self.net_sg = fc(inp_dim=self.obs_dim,
                         arch=pms.sg.arch,
                         acts=pms.sg.acts,
                         lr=pms.sg.lr)

        self.net_cr = fc(inp_dim=self.obs_dim,
                         arch=pms.cr.arch,
                         acts=pms.cr.acts,
                         lr=pms.cr.lr)

        self.mu_epochs = pms.mu.epochs
        self.mu_gen    = pms.mu.gen
        self.mu_batch  = pms.mu.batch

        self.sg_epochs = pms.sg.epochs
        self.sg_gen    = pms.sg.gen
        self.sg_batch  = pms.sg.batch

        self.cr_epochs = pms.cr.epochs
        self.cr_gen    = pms.cr.gen
        self.cr_batch  = pms.cr.batch


        self.obs = 1.0

        self.n_steps_total = self.n_steps_max*self.n_points

        self.summary()

    # Reset
    def reset(self, run):

        # Mother class reset
        super().reset(run)

        # Additional data storage
        self.hist_a = np.zeros((self.n_steps_total)) # advantage

        # Reset networks
        self.net_mu.reset()
        self.net_sg.reset()
        self.net_cr.reset()

        # Initial sampling
        # This fills x array with samples
        return self.sample()

    # Sample from distribution
    def sample(self):

        obs = torch.ones(1,self.dim)*self.obs
        mu  = self.net_mu(obs) + torch.tensor(self.x0)
        sg  = self.net_sg(obs)*torch.tensor(self.sigma0)
        cr  = self.net_cr(obs)

        pdf = self.get_pdf(mu[0], sg[0], cr[0])
        self.x = pdf.sample([self.n_points])

        return self.x

    # Compute full cov pdf
    def get_pdf(self, mu, sg, cr):

        pdf = td.MultivariateNormal(mu.float(), torch.diag(sg.float()))

        return pdf

    # Step
    def step(self, c):

        # Update best point
        self.update_best(c)

        # Store
        self.store(c)

        # Compute advantages
        self.compute_advantages()

        # Update
        self.train_loop(self.mu_epochs, self.mu_gen,
                        self.mu_batch,  self.net_mu)
        self.train_loop(self.sg_epochs, self.sg_gen,
                        self.sg_batch,  self.net_sg)
        self.train_loop(self.cr_epochs, self.cr_gen,
                        self.cr_batch,  self.net_cr)

        # Sample
        self.x = self.sample()

        self.stp += 1

    # Get data history
    def get_history(self, n_gen):

        # Starting and ending indices based on the required nb of generations
        start     = max(0,self.total_stp - n_gen*self.n_points)
        end       = self.total_stp
        buff_size = end - start
        n_gen     = buff_size//self.n_points

        # Randomize batch
        sample = np.arange(start, end)
        np.random.shuffle(sample)

        # Draw elements
        buff_x = torch.from_numpy(self.hist_x[sample])
        buff_a = torch.from_numpy(self.hist_a[sample])

        # Remove elements with zero advantage
        # if (self.adv_clip):
        #     idx    = tf.where(buff_a > 0)
        #     idx    = tf.reshape(idx, [-1])
        #     buff_a = tf.gather(buff_a, idx)
        #     buff_x = tf.gather(buff_x, idx)

        # Reshape
        buff_x = torch.reshape(buff_x, [-1, self.dim])
        buff_a = torch.reshape(buff_a, [-1])

        return buff_x, buff_a

    # Compute advantages
    def compute_advantages(self):

        # Start and end indices of last generation
        start   = max(0,self.total_stp - self.n_points)
        end     = self.total_stp

        # Compute normalized advantage
        avg_rwd = np.mean(self.hist_c[start:end])
        std_rwd = np.std( self.hist_c[start:end])
        adv     = (self.hist_c[start:end] - avg_rwd)/(std_rwd + 1.0e-12)
        adv    *=-1.0

        # Clip advantages if required
        if (self.adv_clip):
            sc = np.argsort(adv)
            adv[sc[:self.n_elite]] = 0.0
            # adv = np.maximum(adv, 0.0)

        # Decay advantage history
        self.hist_a[:] *= self.adv_decay
        self.hist_a[start:end] = adv[:]

    # Train loop
    def train_loop(self, n_epochs, n_gens, n_batch, net):

        # Loop on epochs
        for epoch in range(n_epochs):
            x, a = self.get_history(n_gens)
            done = False
            btc  = 0

            # Visit all available history
            while not done:

                start    = btc*n_batch*self.n_points
                end      = min((btc+1)*n_batch*self.n_points,len(a))
                btc     += 1
                if (end == len(a)): done = True

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
        mu  = self.net_mu(obs) + torch.tensor(self.x0)
        sg  = self.net_sg(obs)*torch.tensor(self.sigma0)
        cr  = self.net_cr(obs)

        pdf = self.get_pdf(mu[0], sg[0], cr[0])
        log = pdf.log_prob(act)

        # Compute loss
        s    = torch.multiply(adv, log)
        loss =-torch.mean(s)

        return loss

    # Compute covariance matrix
    # def get_cov(self, sg, cr):

    #     # # Create skew-symmetric matrix
    #     # t = tf.zeros([self.dim, self.dim])

    #     # idx = 0
    #     # for dg in range(self.dim-1):
    #     #     diag = cr[idx:idx+self.dim-(dg+1)]
    #     #     idx += self.dim-(dg+1)
    #     #     t    = tf.linalg.set_diag(t,  diag, k=-(dg+1))
    #     #     t    = tf.linalg.set_diag(t, -diag, k= (dg+1))

    #     # # Exponentiate to get orthogonal matrix
    #     # et = tf.linalg.expm(t)

    #     # # Generate diagonal matrix
    #     # s = tf.zeros([self.dim, self.dim])
    #     # s = tf.linalg.set_diag(s, sg, k=0)

    #     # # Generate covariance matrix
    #     # cov = tf.matmul(et,s)
    #     # cov = tf.matmul(cov, tf.transpose(et))

    #     # return cov

    #     # Extract sigmas and thetas
    #     sigmas = sg
    #     thetas = cr*math.pi

    #     # Build initial theta matrix
    #     t   = tf.ones([self.dim,self.dim])*math.pi/2.0
    #     t   = tf.linalg.set_diag(t, tf.zeros(self.dim), k=0)
    #     idx = 0
    #     for dg in range(self.dim-1):
    #         diag = thetas[idx:idx+self.dim-(dg+1)]
    #         idx += self.dim-(dg+1)
    #         t    = tf.linalg.set_diag(t, diag, k=-(dg+1))
    #     cor = tf.cos(t)

    #     # Correct upper part to exact zero
    #     for dg in range(self.dim-1):
    #         size = self.dim-(dg+1)
    #         cor  = tf.linalg.set_diag(cor, tf.zeros(size), k=(dg+1))

    #     # Roll and compute additional terms
    #     for roll in range(self.dim-1):
    #         vec = tf.ones([self.dim, 1])
    #         vec = tf.scalar_mul(math.pi/2, vec)
    #         t   = tf.concat([vec, t[:, :self.dim-1]], axis=1)
    #         for dg in range(self.dim-1):
    #             zero = tf.zeros(self.dim-(dg+1))
    #             t    = tf.linalg.set_diag(t, zero, k=dg+1)
    #         cor = tf.multiply(cor, tf.sin(t))

    #     cor = tf.matmul(cor, tf.transpose(cor))
    #     scl = tf.zeros([self.dim, self.dim])
    #     scl = tf.linalg.set_diag(scl, tf.sqrt(sigmas), k=0)
    #     cov = tf.matmul(scl, cor)
    #     cov = tf.matmul(cov, scl)

    #     return cov
