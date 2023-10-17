# Custom imports
from sparkle.src.utils.network import *
from sparkle.src.agent.base import *

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
        self.lr_mu       = 5.0e-3
        self.lr_sg       = 5.0e-3
        self.lr_cr       = 1.0e-3
        self.mu_arch     = [2,2,2]
        self.sg_arch     = [2,2,2]
        self.cr_arch     = [2,2,2]
        self.mu_epochs   = 128
        self.sg_epochs   = 16
        self.cr_epochs   = 16
        self.mu_gen      = 1
        self.sg_gen      = 8
        self.cr_gen      = 16
        self.mu_batch    = 1
        self.sg_batch    = 4
        self.cr_batch    = 8
        self.adv_clip    = True
        self.adv_decay   = 1.0 - math.exp(-0.35*self.dim)
        self.obs_dim     = self.dim
        self.cov_dim     = math.floor(self.dim*(self.dim - 1)/2)

        if hasattr(pms, "n_steps_max"): self.n_steps_max = pms.n_steps_max
        if hasattr(pms, "n_points"):    self.n_points    = pms.n_points
        if hasattr(pms, "obs_dim"):     self.obs_dim     = pms.obs_dim
        if hasattr(pms, "sigma0"):      self.sigma0      = pms.sigma0
        if hasattr(pms, "x0"):          self.x0          = np.array(pms.x0)
        if hasattr(pms, "lr_mu"):       self.lr_mu       = np.array(pms.lr_mu)
        if hasattr(pms, "lr_sg"):       self.lr_sg       = np.array(pms.lr_sg)
        if hasattr(pms, "lr_cr"):       self.lr_cr       = np.array(pms.lr_cr)
        if hasattr(pms, "mu_arch"):     self.mu_arch     = np.array(pms.mu_arch)
        if hasattr(pms, "sg_arch"):     self.sg_arch     = np.array(pms.sg_arch)
        if hasattr(pms, "cr_arch"):     self.cr_arch     = np.array(pms.cr_arch)
        if hasattr(pms, "mu_epochs"):   self.mu_epochs   = np.array(pms.mu_epochs)
        if hasattr(pms, "sg_epochs"):   self.sg_epochs   = np.array(pms.sg_epochs)
        if hasattr(pms, "cr_epochs"):   self.cr_epochs   = np.array(pms.cr_epochs)
        if hasattr(pms, "mu_gen"):      self.mu_gen      = np.array(pms.mu_gen)
        if hasattr(pms, "sg_gen"):      self.sg_gen      = np.array(pms.sg_gen)
        if hasattr(pms, "cr_gen"):      self.cr_gen      = np.array(pms.cr_gen)
        if hasattr(pms, "mu_batch"):    self.mu_batch    = np.array(pms.mu_batch)
        if hasattr(pms, "sg_batch"):    self.sg_batch    = np.array(pms.sg_batch)
        if hasattr(pms, "cr_batch"):    self.cr_batch    = np.array(pms.cr_batch)
        if hasattr(pms, "adv_clip"):    self.adv_clip    = np.array(pms.adv_clip)
        if hasattr(pms, "adv_decay"):   self.adv_decay   = np.array(pms.adv_decay)

        self.net_mu = nn(self.mu_arch, self.obs_dim, self.dim,
                         'tanh', 'tanh',    self.lr_mu)
        self.net_sg = nn(self.sg_arch, self.obs_dim, self.dim,
                         'tanh', 'sigmoid', self.lr_sg)
        self.net_cr = nn(self.cr_arch, self.obs_dim, self.cov_dim,
                         'tanh', 'sigmoid', self.lr_cr)

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
        self.sample()

        return self.x

    # Sample from distribution
    def sample(self):

        # Observation vector
        obs = tf.ones([1,self.dim])*self.obs

        # Predict mu
        x  = tf.convert_to_tensor(obs)
        mu = self.net_mu.call(x)
        mu = np.asarray(mu)[0] + self.x0

        # Predict sigma
        x  = tf.convert_to_tensor(obs)
        sg = self.net_sg.call(x)
        sg = np.asarray(sg)[0]*self.sigma0

        # Predict correlations
        x  = tf.convert_to_tensor(obs)
        cr = self.net_cr.call(x)
        cr = np.asarray(cr)[0]

        # Define pdf
        pdf = self.get_cov_pdf(mu, sg, cr)

        # Draw actions
        self.x = pdf.sample(self.n_points)
        self.x = np.asarray(self.x)

        return self.x

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
        self.sample()

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

        # Draw elements as lists
        buff_x = tf.convert_to_tensor([self.hist_x[i] for i in sample])
        buff_a = tf.convert_to_tensor([self.hist_a[i] for i in sample])

        # Remove elements with zero advantage
        # if (self.adv_clip):
        #     idx    = tf.where(buff_a > 0)
        #     idx    = tf.reshape(idx, [-1])
        #     buff_a = tf.gather(buff_a, idx)
        #     buff_x = tf.gather(buff_x, idx)

        # Reshape
        buff_x = tf.reshape(buff_x, [-1, self.dim])
        buff_a = tf.reshape(buff_a, [-1])

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

                btc_x = x[start:end]
                btc_a = a[start:end]
                btc_o = tf.ones([end-start, self.obs_dim])*self.obs

                self.train(btc_o, btc_a, btc_x, net)

    # Train network
    @tf.function
    def train(self, obs, adv, act, net):

        var = net.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(var)

            cr = tf.convert_to_tensor(self.net_cr.call(obs))
            sg = tf.convert_to_tensor(self.net_sg.call(obs)*self.sigma0)
            mu = tf.convert_to_tensor(self.net_mu.call(obs)+self.x0)

            loss = self.get_loss(obs, adv, act, mu, sg, cr)

        grads = tape.gradient(loss, var)
        norm  = tf.linalg.global_norm(grads)
        net.opt.apply_gradients(zip(grads, var))

    # Compute loss
    def get_loss(self, obs, adv, act, mu, sg, cr):

        # Compute pdf
        pdf = self.get_cov_pdf(mu[0], sg[0], cr[0])
        log = pdf.log_prob(tf.cast(act, tf.float32))

        # Compute loss
        s     = tf.multiply(tf.cast(adv, tf.float32), log)
        loss  =-tf.reduce_mean(s)

        return loss

    # Compute full cov pdf
    def get_cov_pdf(self, mu, sg, cr):

        cov  = self.get_cov(sg, cr)
        scl  = tf.linalg.cholesky(cov)
        pdf  = tfd.MultivariateNormalTriL(tf.cast(mu,  tf.float32),
                                          tf.cast(scl, tf.float32))

        return pdf

    # Compute covariance matrix
    def get_cov(self, sg, cr):

        # # Create skew-symmetric matrix
        # t = tf.zeros([self.dim, self.dim])

        # idx = 0
        # for dg in range(self.dim-1):
        #     diag = cr[idx:idx+self.dim-(dg+1)]
        #     idx += self.dim-(dg+1)
        #     t    = tf.linalg.set_diag(t,  diag, k=-(dg+1))
        #     t    = tf.linalg.set_diag(t, -diag, k= (dg+1))

        # # Exponentiate to get orthogonal matrix
        # et = tf.linalg.expm(t)

        # # Generate diagonal matrix
        # s = tf.zeros([self.dim, self.dim])
        # s = tf.linalg.set_diag(s, sg, k=0)

        # # Generate covariance matrix
        # cov = tf.matmul(et,s)
        # cov = tf.matmul(cov, tf.transpose(et))

        # return cov

        # Extract sigmas and thetas
        sigmas = sg
        thetas = cr*math.pi

        # Build initial theta matrix
        t   = tf.ones([self.dim,self.dim])*math.pi/2.0
        t   = tf.linalg.set_diag(t, tf.zeros(self.dim), k=0)
        idx = 0
        for dg in range(self.dim-1):
            diag = thetas[idx:idx+self.dim-(dg+1)]
            idx += self.dim-(dg+1)
            t    = tf.linalg.set_diag(t, diag, k=-(dg+1))
        cor = tf.cos(t)

        # Correct upper part to exact zero
        for dg in range(self.dim-1):
            size = self.dim-(dg+1)
            cor  = tf.linalg.set_diag(cor, tf.zeros(size), k=(dg+1))

        # Roll and compute additional terms
        for roll in range(self.dim-1):
            vec = tf.ones([self.dim, 1])
            vec = tf.scalar_mul(math.pi/2, vec)
            t   = tf.concat([vec, t[:, :self.dim-1]], axis=1)
            for dg in range(self.dim-1):
                zero = tf.zeros(self.dim-(dg+1))
                t    = tf.linalg.set_diag(t, zero, k=dg+1)
            cor = tf.multiply(cor, tf.sin(t))

        cor = tf.matmul(cor, tf.transpose(cor))
        scl = tf.zeros([self.dim, self.dim])
        scl = tf.linalg.set_diag(scl, tf.sqrt(sigmas), k=0)
        cov = tf.matmul(scl, cor)
        cov = tf.matmul(cov, scl)

        return cov
