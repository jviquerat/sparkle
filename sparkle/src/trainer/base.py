import numpy as np
from numpy import ndarray

from sparkle.src.utils.prints import fmt_float


###############################################
class BaseTrainer():
    """
    Base class for all trainers.

    This class defines the common interface and functionality for all trainers
    used in the optimization framework. It provides methods for resetting,
    optimizing, storing data, dumping data, and printing information about
    the optimization process.
    """
    def __init__(self):
        """
        Initializes the BaseTrainer.
        """

    def optimize(self):
        """
        Performs the optimization process.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    def reset(self, run: int) -> None:
        """
        Resets the trainer for a new run.

        Args:
            run: The run number.
        """

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

    def store_data(self, x: ndarray, c: ndarray) -> None:
        """
        Stores the evaluated data points and their costs.

        This method also updates the best point found so far and keeps track
        of the optimization history.

        Args:
            x: The evaluated points.
            c: The cost values at the evaluated points.
        """

        # Work on copies
        xc = np.copy(x)
        cc = np.copy(c)

        # The update of best points is quite inefficient, but it allows
        # to reproduce historical data when loading a pex or a model
        for i in range(cc.shape[0]):
            if (cc[i] <= self.best_score):
                self.best_score = cc[i]
                self.best_x[:]  = xc[i,:]
                self.best_stp   = self.total_stp

            self.hist_t.append(self.total_stp)
            self.hist_c.append(cc[i])
            self.hist_b.append(self.best_score)
            self.hist_s.append(self.best_stp)
            self.hist_x.append(xc[i,:])

            self.total_stp += 1

    def dump_data(self) -> None:
        """
        Dumps the stored data to a file.

        This method saves the optimization history to a file, including the
        time steps, cost values, best cost values, best steps, and the
        evaluated points.
        """

        filename = self.path+'/raw.dat'
        np.savetxt(filename,
                   np.hstack([np.reshape(np.array(self.hist_t), (-1,1)),
                              np.reshape(np.array(self.hist_c), (-1,1)),
                              np.reshape(np.array(self.hist_b), (-1,1)),
                              np.reshape(np.array(self.hist_s), (-1,1)),
                              np.reshape(np.array(self.hist_x), (-1,self.agent.dim))]),
                   fmt='%.5e')

    def print(self) -> None:
        """
        Prints information about the current optimization step.

        This method prints the current iteration number, the number of
        evaluations, the best score found so far, the iteration at which
        the best score was found, and the best point.
        """

        # Handle no-printing after max step
        if (self.it < self.agent.n_steps_max-1):
            end = "\r"
            self.cnt = 0
        else:
            end  = "\n"
            self.cnt += 1

        # Actual print
        if (self.cnt <= 1):
            gs = fmt_float(self.best_score)
            gb = np.array2string(self.best_x, precision=5, floatmode='fixed',
                                 threshold=4, separator=',')
            blank = "                                                 "
            print("# Iteration #"+str(self.it)+", n_eval = "+str(self.total_stp)+", best score = "+gs+" at individual "+str(self.best_stp)+" for x = "+gb+blank, end=end)
