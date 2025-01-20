# Custom imports
from sparkle.env.base_env import *

###############################################
### Environment for griewank
class griewank(base_env):

    # Create object
    def __init__(self, cpu, path, pms=None):

        # Fill structure
        self.name      = 'griewank'
        self.base_path = path
        self.cpu       = cpu
        self.dim       = 2
        self.x0        = np.array([  5.0,   5.0])
        self.xmin      = np.array([-10.0, -10.0])
        self.xmax      = np.array([ 10.0,  10.0])
        self.it_plt    = 0

        # Check inputs
        if hasattr(pms, "x0"):   self.x0   = pms.x0
        if hasattr(pms, "xmin"): self.xmin = pms.xmin
        if hasattr(pms, "xmax"): self.xmax = pms.xmax

        # Generate map of cost values for rendering
        self.generate_cost_map_2d()

    # Reset environment
    def reset(self, run):

        self.path   = self.base_path+"/"+str(run)
        self.it_plt = 0

        return True

    # Cost function
    def cost(self, x):

        return 1.0 + (x[0]**2+x[1]**2)/4000.0 - math.cos(x[0])*math.cos(x[1]/math.sqrt(2.0))

    # Rendering
    def render(self, x, c):

        if (self.it_plt == 0):
            os.makedirs(self.path+'/png', exist_ok=True)

        plt.clf()
        fig, ax = plt.subplots(figsize=plt.figaspect(self.z))
        ax.set_xlim([self.xmin[0], self.xmax[0]])
        ax.set_ylim([self.xmin[1], self.xmax[1]])
        fig.subplots_adjust(0,0,1,1)
        plt.imshow(self.z,
                   extent=[self.xmin[0], self.xmax[0],
                           self.xmin[1], self.xmax[1]],
                   alpha=0.8, cmap='RdBu_r')
        cnt = plt.contour(self.x, self.y, self.z,
                          levels=[0.1, 1.0, 5.0, 10.0, 20.0],
                          colors='black', alpha=0.5)
        plt.clabel(cnt, inline=True, fontsize=8, fmt="%.0f")
        plt.scatter(x[:,0], x[:,1], c="black", marker='o', alpha=0.8)

        filename = self.path+"/png/"+str(self.it_plt)+".png"
        plt.axis('off')
        plt.savefig(filename, dpi=100)
        plt.close()

        self.it_plt += 1

    # Close environment
    def close(self):
        pass
