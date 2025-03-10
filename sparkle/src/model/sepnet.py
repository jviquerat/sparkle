# Generic imports
import math
import torch
import torch.optim as toptim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Custom imports
from sparkle.src.network.lip_mlp     import lip_mlp
from sparkle.src.optimizer.optimizer import opt_factory
from sparkle.src.env.spaces          import environment_spaces

###############################################
### Separable network model
class sepnet():
    def __init__(self, spaces, pms):


        self.spaces   = spaces
        self.n_epochs = pms.n_epochs

        self.reset()

    # Reset model
    def reset(self):

        # Initialize networks
        self.net = []
        for k in range(self.spaces.dim):
            net = lip_mlp(inp_dim   = 1,
                          out_dim   = 1,
                          arch      = [32,32,32],
                          acts      = ["tanh","tanh","tanh","linear"],
                          lip_const = [2.0, 2.0, 2.0, 5.0])

            self.net.append(net)

        pms_list = []
        for k in range(self.spaces.dim):
            pms_list.append({"params": self.net[k].params()})

        self.opt = toptim.Adam(pms_list, lr=1.0e-3)

    # Evaluate at test points
    def evaluate(self, x):

        xt         = torch.tensor(self.normalize_input(x))
        batch_size = x.shape[0]
        r          = torch.zeros(batch_size, self.spaces.dim)

        for k in range(self.spaces.dim):
            xk     = torch.reshape(xt[:,k], (-1,1))
            r[:,k] = torch.reshape(self.net[k].forward(xk), (batch_size,1))[:,0]

        rr = self.denormalize_output(torch.sum(r, axis=1).detach())

        return rr, np.zeros(batch_size)

    # Build model from input
    def build(self, x, y):

        self.reset()

        self.ymax_ = np.max(y)
        self.ymin_ = np.min(y)
        self.x_    = self.normalize_input(x)
        self.y_    = self.normalize_output(y)

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

            history_loss[epoch,0]   = epoch
            history_loss[epoch,1] /= float(btc)

            end = "\r"
            if (epoch == self.n_epochs-1): end = "\n"
            print("# Epoch #"+str(epoch)+"/"+str(self.n_epochs)+", loss = "+str(history_loss[epoch,1])+"                    ", end=end)

        self.plot_loss("loss.png", history_loss, ["loss"])

    # Compute loss
    # x shape is [batch_size, dim]
    # c shape is [batch_size]
    def get_loss(self, x, c):

        x.requires_grad = True
        batch_size      = x.shape[0]
        r               = torch.zeros(batch_size, self.spaces.dim)

        for k in range(self.spaces.dim):
            xk     = torch.reshape(x[:,k], (-1,1))
            r[:,k] = torch.reshape(self.net[k].forward(xk), (batch_size,1))[:,0]

        r_sum = torch.sum(r, axis=1)
        loss  = torch.mean((c - r_sum)**2)

        return loss

    # Dump model
    def dump(self, filename):
        pass

    # Normalize input
    def normalize_input(self, x):

        xx  = x - 0.5*(self.spaces.xmin + self.spaces.xmax)
        xx /= 0.5*(self.spaces.xmax - self.spaces.xmin)

        return xx

    # Normalize output
    def normalize_output(self, y):

        yy  = y - 0.5*(self.ymin_ + self.ymax_)
        yy /= 0.5*(self.ymax_ - self.ymin_)

        return yy

    # Denormalize output
    def denormalize_output(self, yy):

        y  = yy*0.5*(self.ymax_ - self.ymin_)
        y += 0.5*(self.ymin_ + self.ymax_)

        return y

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

    # Plot function for 2D models only
    def plot_model_2d(self, x_pts, f_cost):

        n_plot = 100
        x      = np.linspace(self.spaces.xmin()[0],
                             self.spaces.xmax()[0], num=n_plot)
        y      = np.linspace(self.spaces.xmax()[1],
                             self.spaces.xmin()[1], num=n_plot)
        grid   = np.array(np.meshgrid(x, y))
        X      = grid.reshape(self.spaces.dim,-1).T
        z      = np.zeros((n_plot, n_plot, 3))

        r, _     = self.evaluate(X)
        rr       = torch.sum(r, axis=1).detach()
        rr       = self.denormalize_output(rr)
        #z[:,:,0] = torch.sum(r, axis=1).detach().reshape(n_plot,n_plot)
        z[:,:,0] = rr.reshape(n_plot,n_plot)
        z[:,:,1] = np.reshape(f_cost(X), (n_plot,n_plot))
        z[:,:,2] = np.abs(z[:,:,1] - z[:,:,0])

        titles = [None]*(3)
        # for k in range(self.r_dim):
        #     titles[k] = "f_"+str(k)+"(x_"+str(k)+")"
        titles[0] = "sum(f_k)"
        titles[1] = "f(x)"
        titles[2] = "f(x) - sum(f_k)"

        plt.clf()
        fig = plt.figure()
        fig.set_size_inches(5*3+3, 5)
        fig.subplots_adjust(0,0,1,1)

        for k in range(3):
            ax = fig.add_subplot(1,3,k+1)
            ax.set_xlim([self.spaces.xmin()[0], self.spaces.xmax()[0]])
            ax.set_ylim([self.spaces.xmin()[1], self.spaces.xmax()[1]])
            ax.axis('off')
            ax.set_title(titles[k])
            ax.imshow(z[:,:,k],
                      extent=[self.spaces.xmin()[0], self.spaces.xmax()[0],
                              self.spaces.xmin()[1], self.spaces.xmax()[1]],
                      vmin=self.spaces.vmin(), vmax=self.spaces.vmax(),
                      alpha=0.8, cmap='RdBu_r')
            if (k == 1):
                ax.scatter(x_pts[:,0], x_pts[:,1], c="black", marker='o', alpha=0.8)
            cnt = ax.contour(x, y, z[:,:,k],
                             levels=self.spaces.levels(),
                             colors='black', alpha=0.5)
            ax.clabel(cnt, inline=True, fontsize=8, fmt="%.0f")

        filename = "rewards.png"
        plt.savefig(filename, dpi=100)
        plt.close()
