# Generic imports
import math
import random
import numpy as np

# Custom imports
from sparkle.src.utils.network import *

###############################################
### PBO
class pbo():
    def __init__(self, path, dim, xmin, xmax, pms):

        self.base_path   = path
        self.dim         = dim
        self.xmin        = xmin
        self.xmax        = xmax

        self.sigma0      = 0.25*(np.min(xmax)-np.max(xmin))
        self.x0          = 0.5*(xmax+xmin)

        self.n_steps_max = 20
        self.n_points    = 10
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
        self.adv_decay   = 0.5
        self.obs_dim     = self.dim
        self.cov_dim     = math.floor(self.dim*(self.dim - 1)/2)

        if hasattr(pms, "n_steps_max"): self.n_steps_max = pms.n_steps_max
        if hasattr(pms, "n_points"):    self.n_points    = pms.n_points
        if hasattr(pms, "obs_dim"):     self.obs_dim     = pms.obs_dim

        # Data storage
        self.n_steps_total = self.n_steps_max*self.n_points
        self.hist_t        = np.zeros((self.n_steps_total))           # time
        self.hist_c        = np.zeros((self.n_steps_total))           # cost
        self.hist_a        = np.zeros((self.n_steps_total))           # advantage
        self.hist_b        = np.zeros((self.n_steps_total))           # best cost
        self.hist_x        = np.zeros((self.n_steps_total, self.dim)) # dofs

    # Reset
    def reset(self, run):

        # Step counter       (one step = lambda cost evaluations)
        # Total step counter (one total step = 1 offspring cost evaluation)
        self.stp = 0
        self.total_stp = 0

        # Best values
        self.best_score = 1.0e8
        self.best_x     = np.zeros(self.dim)

        # Path
        self.path = self.base_path+"/"+str(run)

        # Networks
        self.net_mu = nn(self.mu_arch, self.dim,
                         'relu', 'tanh', self.lr_mu)
        self.net_sg = nn(self.sg_arch, self.dim,
                         'tanh', 'sigmoid', self.lr_sg)
        self.net_cr = nn(self.cr_arch, self.cov_dim,
                         'tanh', 'sigmoid', self.lr_cr)

        # Arrays
        self.obs = tf.ones([1,self.obs_dim])

        # Initialze networks with forward pass
        self.net_mu(self.obs)
        self.net_sg(self.obs)
        self.net_cr(self.obs)

        # Initial sampling
        # This fills x and z arrays with samples
        self.sample()

        return self.x

    # Sample from distribution
    def sample(self):

        # Predict mu
        x  = tf.convert_to_tensor(self.obs)
        mu = self.net_mu.call(x)
        mu = np.asarray(mu)[0] + self.x0

        # Predict sigma
        x  = tf.convert_to_tensor(self.obs)
        sg = self.net_sg.call(x)
        sg = np.asarray(sg)[0]*self.sigma0

        # Predict correlations
        x  = tf.convert_to_tensor(self.obs)
        cr = self.net_cr.call(x)
        cr = np.asarray(cr)[0]

        # Define pdf
        pdf = self.get_cov_pdf(mu, sg, cr)

        # Draw actions
        self.x = pdf.sample(self.n_points)
        self.x = np.asarray(self.x)

        return self.x

    # Update global best
    def update_best(self, c):

        for i in range(self.n_points):
            if (c[i] <= self.best_score):
                self.best_score = c[i]
                self.best_x     = self.x[i,:]

    # Step
    def step(self, c):

        # Update best point
        self.update_best(c)

        # Store
        self.store(c)

        # Compute advantages
        self.compute_advantages()

        # Update
        self.train_loop_sg()
        self.train_loop_cr()
        self.train_loop_mu()

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
        buff_x = [self.hist_x[i] for i in sample]
        buff_a = [self.hist_a[i] for i in sample]

        # Reshape
        buff_x = tf.reshape(buff_x, [buff_size, self.dim])
        buff_a = tf.reshape(buff_a, [buff_size])

        return buff_x, buff_a

    # Compute advantages
    def compute_advantages(self):

        # Start and end indices of last generation
        start   = max(0,self.total_stp - self.n_points)
        end     = self.total_stp# + self.n_points

        # Compute normalized advantage
        avg_rwd = np.mean(self.hist_c[start:end])
        std_rwd = np.std( self.hist_c[start:end])
        adv     = (self.hist_c[start:end] - avg_rwd)/(std_rwd + 1.0e-12)
        adv    *=-1.0

        # Clip advantages if required
        if (self.adv_clip):
            adv = np.maximum(adv, 0.0)

        # Decay advantage history
        self.hist_a[:] *= self.adv_decay
        self.hist_a[start:end] = adv[:]

    # Train loop for mu
    def train_loop_mu(self):

        # Loop on epochs
        for epoch in range(self.mu_epochs):
            n    = self.mu_gen
            x, a = self.get_history(n)
            done = False
            btc  = 0

            # Visit all available history
            while not done:

                start    = btc*self.mu_batch*self.n_points
                end      = min((btc+1)*self.mu_batch*self.n_points,len(a))
                btc     += 1
                if (end == len(a)): done = True

                btc_x = x[start:end]
                btc_a = a[start:end]
                btc_o = tf.ones([end-start, self.obs_dim])

                self.train_mu(btc_o, btc_a, btc_x)

    # Train loop for sg
    def train_loop_sg(self):

        # Loop on epochs
        for epoch in range(self.sg_epochs):
            n    = self.sg_gen
            x, a = self.get_history(n)
            done = False
            btc  = 0

            # Visit all available history
            while not done:

                start    = btc*self.sg_batch*self.n_points
                end      = min((btc+1)*self.sg_batch*self.n_points,len(a))
                btc     += 1
                if (end == len(a)): done = True

                btc_x = x[start:end]
                btc_a = a[start:end]
                btc_o = tf.ones([end-start, self.obs_dim])

                self.train_sg(btc_o, btc_a, btc_x)

    # Train loop for cr
    def train_loop_cr(self):

        # Loop on epochs
        for epoch in range(self.cr_epochs):
            n    = self.cr_gen
            x, a = self.get_history(n)
            done = False
            btc  = 0

            # Visit all available history
            while not done:

                start    = btc*self.cr_batch*self.n_points
                end      = min((btc+1)*self.cr_batch*self.n_points,len(a))

                btc     += 1
                if (end == len(a)): done = True

                btc_x = x[start:end]
                btc_a = a[start:end]
                btc_o = tf.ones([end-start, self.obs_dim])

                self.train_cr(btc_o, btc_a, btc_x)

    # Train mu network
    @tf.function
    def train_mu(self, obs, adv, act):

        var = self.net_mu.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(var)

            cr = tf.convert_to_tensor(self.net_cr.call(obs))
            sg = tf.convert_to_tensor(self.net_sg.call(obs)*self.sigma0)
            mu = tf.convert_to_tensor(self.net_mu.call(obs)+self.x0)

            loss = self.get_loss(obs, adv, act, mu, sg, cr)

        grads = tape.gradient(loss, var)
        norm  = tf.linalg.global_norm(grads)
        self.net_mu.opt.apply_gradients(zip(grads, var))

    # Train sg network
    @tf.function
    def train_sg(self, obs, adv, act):

        var = self.net_sg.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(var)

            cr = tf.convert_to_tensor(self.net_cr.call(obs))
            sg = tf.convert_to_tensor(self.net_sg.call(obs)*self.sigma0)
            mu = tf.convert_to_tensor(self.net_mu.call(obs)+self.x0)

            loss = self.get_loss(obs, adv, act, mu, sg, cr)

        grads = tape.gradient(loss, var)
        norm  = tf.linalg.global_norm(grads)
        self.net_sg.opt.apply_gradients(zip(grads, var))

    # Train cr network
    @tf.function
    def train_cr(self, obs, adv, act):

        var = self.net_cr.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(var)

            cr = tf.convert_to_tensor(self.net_cr.call(obs))
            sg = tf.convert_to_tensor(self.net_sg.call(obs)*self.sigma0)
            mu = tf.convert_to_tensor(self.net_mu.call(obs)+self.x0)

            loss = self.get_loss(obs, adv, act, mu, sg, cr)

        grads = tape.gradient(loss, var)
        norm  = tf.linalg.global_norm(grads)
        self.net_cr.opt.apply_gradients(zip(grads, var))

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

        # Extract sigmas and thetas
        #sigmas = 0.85*sg
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
        scl = tf.linalg.set_diag(scl, sigmas, k=0)
        cov = tf.matmul(scl, cor)
        cov = tf.matmul(cov, scl)

        return cov

    # Return degrees of freedom
    def dof(self):

        return self.x

    # Return number of degress of freedom
    def ndof(self):

        return self.n_points

    # Check if done
    def done(self):

        if (self.stp == self.n_steps_max):
            return True

        return False

    # Store data
    def store(self, c):

        for i in range(self.n_points):
            self.hist_t[self.total_stp]   = self.total_stp
            self.hist_x[self.total_stp,:] = self.x[i,:]
            self.hist_c[self.total_stp]   = c[i]
            self.hist_b[self.total_stp]   = self.best_score

            self.total_stp += 1

    # Dump data
    def dump(self):

        filename = self.path+'/raw.dat'
        np.savetxt(filename,
                   np.hstack([np.reshape(self.hist_t, (-1,1)),
                              np.reshape(self.hist_c, (-1,1)),
                              np.reshape(self.hist_b, (-1,1)),
                              np.reshape(self.hist_x, (-1,self.dim))]),
                   fmt='%.5e')

    # Print
    def print(self):

        # Total nb of evaluations
        n_eval = self.stp*self.n_points

        # Handle no-printing after max step
        if (self.stp < self.n_steps_max-1):
            end = "\r"
            self.cnt = 0
        else:
            end  = "\n"
            self.cnt += 1

        # Actual print
        if (self.cnt <= 1):
            gs = f"{self.best_score:.5e}"
            gb = np.array2string(self.best_x, precision=5, floatmode='fixed',
                                 threshold=5, separator=',')
            print("# Step #"+str(self.stp)+", n_eval = "+str(n_eval)+", best score = "+str(gs)+" at x = "+str(gb)+"                                                                                   ", end=end)

