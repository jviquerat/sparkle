import numpy as np
from numpy import ndarray


###############################################
### Exponential moving average class
class EMA:
    def __init__(self, alpha: float, n: int) -> None:
        self.alpha = alpha
        self.n     = n

    ### Smooth entire array
    def smooth(self, array: ndarray) -> ndarray:

        s    = np.zeros_like(array)
        y    = array[0]
        s[0] = y

        for i in range(1,len(array)):
            v    = array[i]
            s[i] = self.alpha*v + (1.0-self.alpha)*y
            y    = np.mean(array[max(i-self.n,0):i])

        return s

###############################################
### Report buffer, used to store learning metrics
# frequency : writing frequency (in number of calls to write() )
# names     : dict names for report
class Report:
    def __init__(self, frequency, names):

        self.frequency = frequency
        self.names     = names
        self.reset()

    # Reset
    def reset(self):

        self.count = 0
        self.data  = {}
        for name in self.names:
            self.data[name] = []

    # Append data to the report
    def append(self, name, value):

        self.data[name].append(value)

    # Get data from the report
    def get(self, name):

        return self.data[name]

    # Return an average of n last values of given field
    def avg(self, name, n):

        return np.mean(self.data[name][-n:])

    # Write report
    def write(self, filename, force=False):

        self.count += 1
        if ((self.count%self.frequency == 0) or (force == True)):

            # Generate array to save
            array = np.array([])
            for name in self.names:
                tmp = np.array(self.data[name],dtype=float)
                if array.size: array = np.vstack((array, tmp))
                else:          array = tmp

            array = np.transpose(array)
            array = np.nan_to_num(array, nan=0.0)

            # Save as a csv file
            np.savetxt(filename, array, fmt='%.5e')

            # Reset
            self.count = 0
