# Generic imports
import math

# Custom imports
from sparkle.env.base_env    import *
from sparkle.src.utils.lsrk4 import *

###############################################
### Environment for lorenz control
class lorenz(base_env):

    # Create object
    def __init__(self, cpu, path, pms=None):

        # Fill structure
        self.name      = 'lorenz'
        self.base_path = path
        self.cpu       = cpu
        self.dim       = 4
        self.xmin      = np.array([-1.0,-1.0,-1.0,-1.0])
        self.xmax      = np.array([ 1.0, 1.0, 1.0, 1.0])
        self.it_plt    = 0

        self.sigma     = 10.0
        self.rho       = 28.0
        self.beta      = 8.0/3.0
        self.t_max     = 25.0
        self.dt        = 0.005
        self.n_steps   = math.floor(self.t_max/self.dt)

        self.gif_steps = 300

        # Check inputs
        if hasattr(pms, "xmin"): self.xmin = pms.xmin
        if hasattr(pms, "xmax"): self.xmax = pms.xmax

        # Arrays
        self.x  = np.zeros(3)                   # unknowns
        self.xk = np.zeros(3)                   # lsrk storage
        self.fx = np.zeros(3)                   # rhs
        self.hx = np.zeros((self.n_steps+1, 3)) # time storage
        self.t  = np.linspace(0.0, self.t_max, num=self.n_steps+1)

        # Initialize integrator
        self.integrator = lsrk4()

    # Reset environment
    def reset(self, run):

        self.path   = self.base_path+"/"+str(run)
        self.it_plt = 0

        return True

    # Cost function
    def cost(self, u):

        # Initializations
        self.x[0]    = 10.0
        self.x[1]    = 10.0
        self.x[2]    = 10.0
        self.hx[0,:] = self.x[:]

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
            if (self.x[0] < 0.0): v -= self.dt

        return v

    # Rendering
    def render(self, x, c, pms=None):

        if (self.it_plt == 0):
            os.makedirs(self.path+'/png', exist_ok=True)

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

        self.it_plt += 1

    # Rendering with gif
    def render_gif(self, x):

        os.makedirs(self.path+"/lorenz_gif", exist_ok=True)

        stp = math.floor(self.n_steps/self.gif_steps)
        for i in range(self.gif_steps):
            plt.clf()
            plt.cla()
            fig = plt.figure(tight_layout=True)
            ax  = fig.add_subplot(projection='3d')
            ax.set_axis_off()
            ax.set_xlim([-20.0, 20.0])
            ax.set_ylim([-20.0, 20.0])
            ax.set_zlim([  0.0, 40.0])
            j = i*stp
            ax.plot(self.hx[:j,0], self.hx[:j,1], self.hx[:j,2],
                    linewidth=1)

            filename = self.path+"/lorenz_gif/"+str(i)+".png"
            bbox = fig.bbox_inches.from_bounds(1, 1.25, 4.75, 3)
            plt.savefig(filename, dpi=100, bbox_inches=bbox)
            plt.close()

    # Close environment
    def close(self):
        pass
