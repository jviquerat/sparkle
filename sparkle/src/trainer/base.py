# Generic imports
import os
import math
import shutil
import matplotlib
import matplotlib.pyplot as plt
import numpy             as np

###############################################
### Base trainer
class base_trainer():
    def __init__(self):
        pass

    # Optimize
    def optimize(self):
        raise NotImplementedError

    # Reset
    def reset(self, run):

        # Step counters
        self.total_stp = 0
        self.it_plt = 0

        # Data storage
        self.hist_t = [] # time
        self.hist_c = [] # cost
        self.hist_b = [] # best cost
        self.hist_s = [] # best step
        self.hist_x = [] # dofs

        # Best point
        self.best_x     = np.zeros(self.agent.dim)
        self.best_score = 1.0e15
        self.best_stp   =-1

        # Path
        self.path = self.base_path+"/"+str(run)

    # Store data
    def store_data(self, x, c):

        # The update of best points is quite inefficient, but it allows
        # to reproduce historical data when loading a pex or a model
        for i in range(c.shape[0]):
            if (c[i] <= self.best_score):
                self.best_score = c[i]
                self.best_x[:]  = x[i,:]
                self.best_stp   = self.total_stp

            self.hist_t.append(self.total_stp)
            self.hist_c.append(c[i])
            self.hist_b.append(self.best_score)
            self.hist_s.append(self.best_stp)
            self.hist_x.append(x[i,:])

            self.total_stp += 1

    # Dump data
    def dump_data(self):

        filename = self.path+'/raw.dat'
        np.savetxt(filename,
                   np.hstack([np.reshape(np.array(self.hist_t), (-1,1)),
                              np.reshape(np.array(self.hist_c), (-1,1)),
                              np.reshape(np.array(self.hist_b), (-1,1)),
                              np.reshape(np.array(self.hist_s), (-1,1)),
                              np.reshape(np.array(self.hist_x), (-1,self.agent.dim))]),
                   fmt='%.5e')

    # Print
    def print(self):

        # Handle no-printing after max step
        if (self.it < self.agent.n_steps_max-1):
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
            print("# Iteration #"+str(self.it)+", n_eval = "+str(self.total_stp)+", best score = "+str(gs)+" at individual "+str(self.best_stp)+" for x = "+str(gb)+"                                                                                                           ", end=end)

    # Generate cost map for rendering 1D or 2D envs
    def generate_cost_map(self):

        if (self.env.spaces.dim == 1):
            self.n_plot   = 100
            self.x_plot   = np.linspace(self.env.spaces.xmin[0], self.env.spaces.xmax[0], num=self.n_plot)
            self.cost_map = np.zeros(self.n_plot)

            for i in range(self.n_plot):
                x = np.array([[self.x_plot[i]]])
                self.cost_map[i] = self.env.cost(x)

        if (self.env.spaces.dim == 2):
            self.n_plot   = 100
            self.x_plot   = np.linspace(self.env.spaces.xmin[0], self.env.spaces.xmax[0], num=self.n_plot)
            self.y_plot   = np.linspace(self.env.spaces.xmax[1], self.env.spaces.xmin[1], num=self.n_plot)
            grid          = np.array(np.meshgrid(self.x_plot, self.y_plot))
            self.x_plot   = grid[0]
            self.y_plot   = grid[1]
            self.cost_map = np.zeros((self.n_plot,self.n_plot))

            for i in range(self.n_plot):
                for j in range(self.n_plot):
                    x = np.array([[self.x_plot[i,j], self.y_plot[i,j]]])
                    self.cost_map[i,j] = self.env.cost(x)[0]
