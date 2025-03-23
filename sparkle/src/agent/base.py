# Generic imports

# Custom imports
from sparkle.src.utils.prints import spacer

###############################################
### Base agent
class base_agent():
    def __init__(self, path, spaces, pms):

        self.spaces    = spaces
        self.base_path = path

        self.silent = False
        if hasattr(pms, "silent"): self.silent = pms.silent

    @property
    def dim(self):
        return self.spaces.dim

    @property
    def natural_dim(self):
        return self.spaces.natural_dim

    @property
    def x0(self):
        return self.spaces.x0

    @property
    def xmin(self):
        return self.spaces.xmin

    @property
    def xmax(self):
        return self.spaces.xmax

    # Reset
    def reset(self, run):

        # Step counter (one step = n_points cost evaluations)
        self.stp = 0

        # Path
        self.path = self.base_path+"/"+str(run)

    # Sample
    def sample(self):
        raise NotImplementedError

    # Perform one optimization step
    def step(self, c):
        raise NotImplementedError

    # Render
    def render(self):
        raise NotImplementedError

    # Print informations
    def summary(self):

        spacer("Using "+self.name+" algorithm with "+str(self.n_points)+" points")
        spacer("Problem dimensionality is "+str(self.dim))

    # Return number of degress of freedom
    def ndof(self):

        return self.n_points

    # Check if done
    def done(self):

        if (self.stp == self.n_steps_max): return True
        return False
