# Generic imports
import math
import random
import numpy as np

# Custom imports
from sparkle.src.utils.prints import *

###############################################
### Base agent
class base_agent():
    def __init__(self):
        pass

    # Reset
    def reset(self, run):
        raise NotImplementedError

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

        spacer()
        print("Using "+self.name+" algorithm with "+str(self.n_points)+" points")
        spacer()
        print("Problem dimensionality is "+str(self.dim))

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
                                 threshold=4, separator=',')
            print("# Step #"+str(self.stp)+", n_eval = "+str(n_eval)+", best score = "+str(gs)+" at x = "+str(gb)+"                                                                                                           ", end=end)
