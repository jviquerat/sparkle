# Generic imports
import os
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy             as np

###############################################
### Base environment
class base_env():

    # Create object
    def __init__(self):
        pass

    # Reset environment
    def reset(self):
        raise NotImplementedError

    # Cost function
    def cost(self, x):
        raise NotImplementedError

    # Rendering
    def render(self):
        raise NotImplementedError

    # Close environment
    def close(self):
        raise NotImplementedError

    # Generate cost map for rendering of 2D envs
    def generate_cost_map_2d(self):

        self.n_plot    = 100
        self.x         = np.linspace(self.xmin[0], self.xmax[0], num=self.n_plot)
        self.y         = np.linspace(self.xmax[1], self.xmin[1], num=self.n_plot)
        self.x, self.y = np.array(np.meshgrid(self.x, self.y))
        self.z         = np.zeros((self.n_plot,self.n_plot))

        for i in range(self.n_plot):
            for j in range(self.n_plot):
                self.z[i,j] = self.cost([self.x[i,j], self.y[i,j]])

    # Render 1D envs
    def render_1d(self, x, x_mu=None, y_mu=None, y_std=None, ei=None, x_ei=None):

        if (self.it_plt == 0):
            os.makedirs(self.path+'/png', exist_ok=True)

        # Check inputs
        plot_estimates = False
        if ((y_mu  is not None) and
            (y_std is not None) and
            (ei    is not None) and
            (x_ei  is not None)): plot_estimates = True

        # Set number of subplots
        plt.clf()
        fig = plt.figure()
        if (plot_estimates):
            ax = fig.add_subplot(211)
        else:
            ax = fig.add_subplot(111)

        # Plot function with sampling points
        ax.set_xlim([self.xmin[0], self.xmax[0]])
        ax.set_ylim([-2.5, 3.0])
        ax.grid()
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.plot(self.x_plot, self.y_plot, label="f(x)")
        ax.set_ylabel('y')

        y = np.zeros_like(x)
        for i in range(len(x)):
            y[i] = self.cost(x[i])

        ax.scatter(x[:,0], y[:,0], c="black", marker='o', alpha=0.8,
                   label="observations")
        ax.legend(loc='upper left')

        if (plot_estimates):
            ax.plot(x_mu, y_mu, linestyle='dashed', label="model")
            ax.fill_between(x_mu, y_mu-y_std, y_mu+y_std, alpha=0.2,
                            label="confidence interval")

            ax = fig.add_subplot(212)
            ax.set_xlim([self.xmin[0], self.xmax[0]])
            ax.plot(x_mu, -ei, color='r')
            ax.grid()
            ax.set_yticklabels([])
            ax.set_yticks([])
            ax.set_ylabel('expected improvement')

        # Save figure
        filename = self.path+"/png/"+str(self.it_plt)+".png"
        fig.tight_layout()
        plt.savefig(filename, dpi=100)
        plt.close()

        self.it_plt += 1

    # Render 2D envs
    def render_2d(self, x, vmin, vmax, levels, y_mu=None, y_std=None, x_ei=None):

        if (self.it_plt == 0):
            os.makedirs(self.path+'/png', exist_ok=True)

        # Check inputs
        plot_estimates = False
        if ((y_mu  is not None) and
            (y_std is not None) and
            (x_ei  is not None)): plot_estimates = True

        # Set number of subplots
        plt.clf()
        fig = plt.figure()
        if plot_estimates:
            ax = fig.add_subplot(131)
            fig.set_size_inches(8, 3)
            fig.tight_layout()
            plt.subplots_adjust(bottom=0.01, top=0.99, left=0.01, right=0.99)
            plt.rcParams.update({'axes.titlesize': 'small'})
        else:
            ax = fig.add_subplot(111)
            fig.set_size_inches(3, 3)
            fig.subplots_adjust(0,0,1,1)

        # Plot function with sampling points
        ax.set_xlim([self.xmin[0], self.xmax[0]])
        ax.set_ylim([self.xmin[1], self.xmax[1]])
        ax.axis('off')
        ax.imshow(self.z,
                  extent=[self.xmin[0], self.xmax[0],
                          self.xmin[1], self.xmax[1]],
                  vmin=vmin, vmax=vmax,
                  alpha=0.8, cmap='RdBu_r')
        cnt = ax.contour(self.x, self.y, self.z,
                         levels=levels,
                         colors='black', alpha=0.5)
        ax.clabel(cnt, inline=True, fontsize=8, fmt="%.0f")
        ax.scatter(x[:,0], x[:,1], c="black", marker='o', alpha=0.8)
        if plot_estimates: ax.set_title("f(x)")

        if plot_estimates:
            # Plot y_mu
            ax = fig.add_subplot(132)
            ax.axis('off')
            ax.imshow(y_mu,
                      extent=[self.xmin[0], self.xmax[0],
                              self.xmin[1], self.xmax[1]],
                      vmin=vmin, vmax=vmax,
                      alpha=0.8, cmap='RdBu_r')
            cnt = ax.contour(self.x, self.y, y_mu,
                             levels=levels,
                             colors='black', alpha=0.5)
            ax.clabel(cnt, inline=True, fontsize=8, fmt="%.0f")
            ax.scatter(x_ei[0], x_ei[1], c='red', marker='o', alpha=0.8)
            ax.set_title("model")

            # Plot y_std
            ax = fig.add_subplot(133)
            ax.axis('off')
            ax.imshow(y_std,
                      extent=[self.xmin[0], self.xmax[0],
                              self.xmin[1], self.xmax[1]],
                      alpha=0.8, cmap='RdBu_r')
            cnt = ax.contour(self.x, self.y, y_mu,
                             levels=levels,
                             colors='black', alpha=0.5)
            ax.clabel(cnt, inline=True, fontsize=8, fmt="%.0f")
            ax.set_title("confidence interval")

        # Save figure
        filename = self.path+"/png/"+str(self.it_plt)+".png"
        plt.savefig(filename, dpi=100)
        plt.close()

        self.it_plt += 1
