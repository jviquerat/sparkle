# Generic imports
import math
import torch
from torch.autograd import grad as autograd
import torch.optim as toptim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Custom imports
from sparkle.src.model.base          import base_model
from sparkle.src.network.lip_mlp     import lip_mlp
from sparkle.src.optimizer.optimizer import opt_factory

###############################################
### Lipschitz network model
class lipnet(base_model):
    def __init__(self, spaces, path, pms):
        super().__init__(spaces, path)

        self.n_epochs = pms.n_epochs

        self.reset()

    # Reset model
    def reset(self):

        # Initialize network
        self.net = lip_mlp(inp_dim   = self.spaces.dim,
                           out_dim   = 1,
                           arch      = [32,32],
                           acts      = ["tanh","tanh","linear"],
                           lip_const = [2.0, 2.0, 2.0])

        self.opt = toptim.Adam(self.net.params(), lr=2.0e-3)

    # Evaluate at test points
    def evaluate(self, x):

        # x shape is (n_batch, input_dim)
        n_batch  = x.shape[0]
        y_out    = np.zeros(n_batch)
        lip      = np.zeros(n_batch)
        grad_lip = np.zeros(n_batch)

        # Normalize input
        xx = self.normalize(x, self.spaces.xmin, self.spaces.xmax)

        # Loop on inputs and compute grads
        for k in range(n_batch):
            xt = torch.tensor(xx[k]).requires_grad_(True)
            y  = self.net.forward(xt)

            grad_outputs = torch.ones_like(y)
            grad = autograd(outputs=y,
                            inputs=xt,
                            create_graph=True,
                            grad_outputs=grad_outputs)[0]
            local_lip = grad.norm()

            grad = autograd(outputs=local_lip,
                            inputs=xt,
                            create_graph=False)[0]
            local_grad_lip = grad.norm()

            lip[k]      = local_lip.item()
            grad_lip[k] = local_grad_lip.item()

            y_out[k]     = y.detach().item()

        sample_density = self.gaussian_kde(xx, self.x_, bandwidth=0.2)
        sample_density /= np.max(sample_density)

        # Denormalize evaluation
        y  = self.denormalize(y_out, self.ymin_, self.ymax_)

        #imp = np.maximum(self.ymin_ - y, 0.0)
        imp = self.sigmoid(self.ymin_ - y)

        acq = imp*grad_lip/sample_density
        #acq = np.log(acq)

        return y, acq

    def sigmoid(self, x):

        return 1.0/(1.0 + np.exp(-x))

    # Build model from input
    def build(self, x, y):

        self.reset()

        # Normalization steps
        self.ymax_ = np.max(y)
        self.ymin_ = np.min(y)
        self.x_    = self.normalize(x, self.spaces.xmin, self.spaces.xmax)
        self.y_    = self.normalize(y, self.ymin_, self.ymax_)

        n_points     = x.shape[0]
        batch_size   = math.floor(1.0*n_points)
        history_loss = np.zeros((self.n_epochs, 2))

        for epoch in range(self.n_epochs):

            r = np.arange(0, n_points)
            p = np.random.permutation(r)

            xb = torch.from_numpy(self.x_[p])
            c  = torch.from_numpy(self.y_[p])

            done = False
            btc  = 0

            while not done:
                start = btc*batch_size
                end   = min((btc+1)*batch_size, n_points)
                btc  += 1
                if (end == n_points): done = True

                loss = self.get_loss(xb[start:end], c[start:end])

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                history_loss[epoch,1] += loss

            history_loss[epoch,0]  = epoch
            history_loss[epoch,1] /= float(btc)

            end = "\r"
            if (epoch == self.n_epochs-1): end = "\n"
            print("# Epoch #"+str(epoch)+"/"+str(self.n_epochs)+", loss = "+str(history_loss[epoch,1])+"                    ", end=end)

        self.plot_loss(self.path+"/loss.png", history_loss, ["loss"])

        print("# Lipschitz constants: "+str(self.net.lip_consts()))

    # Compute loss
    # x shape is [batch_size, dim]
    # c shape is [batch_size]
    def get_loss(self, x, c):

        r    = self.net.forward(x.requires_grad_(True))
        loss = torch.mean((c - r.reshape(-1))**2)

        return loss

    # Compute gaussien kernel density
    def gaussian_kde(self, eval_pts, sample_pts, bandwidth=0.5):
        """
        Estimate the density at given evaluation points using Gaussian KDE.

        Parameters:
        - evaluation_points: array of shape (n, d)
        Points at which to evaluate the density.
        - sample_points: array of shape (m, d)
        Data points used to build the density estimate.
        - bandwidth: float
        The bandwidth parameter (h) controlling the smoothness.

        Returns:
        - density: array of shape (n,)
        Density estimates at the evaluation points.
        """
        n, d = eval_pts.shape
        m = sample_pts.shape[0]

        # Calculate the difference between each evaluation point and each sample point
        # This creates an array of shape (n, m, d)
        diff = eval_pts[:, np.newaxis, :] - sample_pts[np.newaxis, :, :]

        # Compute the squared Euclidean distance for each difference vector
        # Resulting in an array of shape (n, m)
        squared_distances = np.sum(diff**2, axis=2)

        # Calculate the normalization factor for the Gaussian kernel in d dimensions
        norm_factor = (2 * np.pi) ** (d / 2) * bandwidth ** d

        # Compute the Gaussian kernel values for each pair (evaluation_point, sample_point)
        kernel_values = np.exp(-0.5 * squared_distances / (bandwidth ** 2)) / norm_factor

        # Average over all sample points to get the density estimate at each evaluation point
        density = np.mean(kernel_values, axis=1)

        return density

    # Dump model
    def dump(self, filename="lipnet.dat"):
        pass

    # Plot losses
    # loss is a np.array of shape (n_samples, n_losses+1)
    # the first column is the epoch number
    def plot_loss(self, filename, loss, labels):

        plt.clf()
        fig = plt.figure()

        for k in range(loss.shape[1]-1):
            plt.plot(loss[:,0], loss[:,k+1], label=labels[k])

        plt.yscale('log')
        plt.legend(loc="upper right")
        plt.savefig(filename, dpi=100)
        plt.close()
