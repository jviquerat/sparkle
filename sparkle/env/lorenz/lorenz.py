import os
import math
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

from sparkle.env.base_env import base_env
from sparkle.src.utils.default import set_default


###############################################
class lorenz(base_env):
    """
    Lorenz attractor problem with a forcing term:

    x' = sigma*(y - x)
    y' = x*(rho - z) - y + u
    z' = x*y - beta*z

    where u = tanh(a*x' + b*y' + c*z' + d)

    Free parameters are (a,b,c,d), the objective being to constrain
    the attractor in the proximity of the x<0 critical point
    """
    def __init__(self, cpu, path, pms=None):

        # Fill structure
        self.name        = 'lorenz'
        self.base_path   = path
        self.cpu         = cpu
        self.render_mode = set_default("render_mode", "png", pms)

        self.dim  = 4
        self.x0   = np.zeros(self.dim)
        self.xmin =-np.ones(self.dim)
        self.xmax = np.ones(self.dim)

        # Plotting data
        self.it_plt    = 0
        self.gif_steps = 500

        # Physical and numerical parameters
        self.sigma   = set_default("sigma", 10.0, pms)
        self.rho     = set_default("rho", 28.0, pms)
        self.beta    = set_default("beta", 8.0/3.0, pms)
        self.t_max   = set_default("t_max", 25.0, pms)
        self.dt      = 0.005
        self.n_steps = math.floor(self.t_max/self.dt)

        # Arrays
        self.x  = np.zeros(3)                   # unknowns
        self.xk = np.zeros(3)                   # lsrk storage
        self.fx = np.zeros(3)                   # rhs
        self.hx = np.zeros((self.n_steps+1, 3)) # time storage
        self.t  = np.linspace(0.0, self.t_max, num=self.n_steps+1)

        # Initialize integrator
        self.integrator = lsrk4()

    def reset(self, run):
        """
        Resets the environment for a new run
        """
        self.path   = self.base_path+"/"+str(run)
        self.it_plt = 0

        os.makedirs(self.path, exist_ok=True)
        os.makedirs(os.path.join(self.path, self.render_mode), exist_ok=True)

        return True

    def cost(self, u):
        """
        Runs the simulation and compute cost function
        """
        # Initializations
        self.x[0]    = 10.0
        self.x[1]    = 10.0
        self.x[2]    = 10.0
        self.hx[0,:] = self.x[:]

        # Target critical point
        x_eq = np.array([-math.sqrt(self.beta*(self.rho - 1.0)),
                         -math.sqrt(self.beta*(self.rho - 1.0)),
                         self.rho - 1.0])

        v = 0.0
        i = 0

        while (i < self.n_steps):

            # Update intermediate storage
            self.xk[:] = self.x[:]

            # Loop on integration steps
            for j in range(self.integrator.steps()):

                # Compute rhs without forcing term
                self.fx[0] = self.sigma*(self.xk[1] - self.xk[0])
                self.fx[1] = self.xk[0]*(self.rho - self.xk[2]) - self.xk[1]
                self.fx[2] = self.xk[0]*self.xk[1] - self.beta*self.xk[2]

                # Compute forcing term
                fu = math.tanh(u[0]*self.fx[0]/10.0 +
                               u[1]*self.fx[1]/20.0 +
                               u[2]*self.fx[2]/40.0 + u[3])

                # Add forcing term
                self.fx[1] += fu

                # Update
                self.integrator.update(self.x, self.xk, self.fx, j, self.dt)

            # Update main storage
            self.x[:] = self.xk[:]

            # Update loop index
            i += 1

            # Store unknowns
            self.hx[i,:] = self.x[:]

            # Update cost
            v += np.linalg.norm(self.x - x_eq)*self.dt

        return v

    def render(self, x, c):
        """
        Renders the trajectory of the best candidate based on self.render_mode
        """
        # Evaluate best candidate
        u = x[np.argmin(c)]
        _ = self.cost(u)

        # png mode is used during training
        if self.render_mode == 'png':
            plt.clf()
            plt.cla()
            fig, ax = plt.subplots(figsize=(8,2))
            fig.tight_layout()
            plt.plot(self.t, self.hx[:,0])
            ax.set_xlim([0.0, self.t_max])
            ax.set_ylim([-20.0, 20.0])

            filename = self.path+"/png/"+str(self.it_plt)+".png"
            plt.grid()
            plt.savefig(filename, dpi=100)
            plt.close()

        # gif mode is used for rendering
        elif self.render_mode == 'gif':

            # Compute plotting limits
            all_x = self.hx[:, 0].flatten()
            all_y = self.hx[:, 1].flatten()
            all_z = self.hx[:, 2].flatten()
            xmin = np.min(all_x)
            xmax = np.max(all_x)
            ymin = np.min(all_y)
            ymax = np.max(all_y)
            zmin = np.min(all_z)
            zmax = np.max(all_z)

            with mplstyle.context('dark_background'):
                colors = plt.cm.RdBu(np.linspace(0, 1, 2))
                stp = math.floor(self.n_steps/self.gif_steps)
                for i in range(self.gif_steps):
                    plt.clf()
                    plt.cla()
                    fig = plt.figure(tight_layout=True)
                    ax  = fig.add_subplot(projection='3d', facecolor='black')

                    ax.set_xlim([xmin, xmax])
                    ax.set_ylim([ymin, ymax])
                    ax.set_zlim([zmin, zmax])
                    ax.axis('off')

                    j = i*stp
                    ax.plot(self.hx[:j,0], self.hx[:j,1], self.hx[:j,2],
                            color='w', linewidth=1.0)
                    ax.scatter(self.hx[j,0], self.hx[j,1], self.hx[j,2],
                               marker='o', s=100, color=colors[1],
                               edgecolor='w', linewidth=0.5)

                    filename = os.path.join(self.path, self.render_mode)
                    filename = os.path.join(filename, f"{i}.png")
                    plt.savefig(filename,
                                dpi=100,
                                facecolor='black',
                                bbox_inches='tight',
                                pad_inches=0.0)
                    plt.close()

        self.it_plt += 1

    def close(self):
        pass

###############################################
class lsrk4():
    """
    Five-stage fourth-order low-storage Runge-Kutta class
    """
    def __init__(self):

        # lsrk coefficients
        self.n_lsrk = 5
        self.a = np.array([ 0.000000000000000, -0.417890474499852,
                           -1.192151694642677, -1.697784692471528,
                           -1.514183444257156], dtype=np.float64)
        self.b = np.array([ 0.149659021999229,  0.379210312999627,
                            0.822955029386982,  0.699450455949122,
                            0.153057247968152], dtype=np.float64)
        self.c = np.array([ 0.000000000000000,  0.149659021999229,
                            0.370400957364205,  0.622255763134443,
                            0.958282130674690], dtype=np.float64)

    # return number of integration steps
    def steps(self):

        return self.n_lsrk

    # return source time at jth step
    def source_time(self, j, t, dt):

        return t + self.c[j]*dt

    # lsrk update
    def update(self, u, uk, f, j, dt):

        for i in range(len(u)):
            u[i]   = self.a[j]*u[i] + dt*f[i]
            uk[i] += self.b[j]*u[i]
