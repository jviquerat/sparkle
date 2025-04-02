import numpy as np
from numpy import ndarray


class EMA:
    """
    Exponential Moving Average (EMA) class.

    This class provides methods for calculating the exponential moving average
    of a data array.
    """
    def __init__(self, alpha: float, n: int) -> None:
        """
        Initializes the EMA calculator.

        Args:
            alpha: The smoothing factor (between 0 and 1).
            n: The number of previous values to consider for averaging.
        """
        self.alpha = alpha
        self.n     = n

    def smooth(self, array: ndarray) -> ndarray:
        """
        Applies exponential moving average smoothing to an array.

        Args:
            array: The input array to smooth.

        Returns:
            A new array containing the smoothed values.
        """

        s    = np.zeros_like(array)
        y    = array[0]
        s[0] = y

        for i in range(1,len(array)):
            v    = array[i]
            s[i] = self.alpha*v + (1.0-self.alpha)*y
            y    = np.mean(array[max(i-self.n,0):i])

        return s

class Report:
    """
    Report buffer class.

    This class provides methods for storing and writing learning metrics.
    """
    def __init__(self, frequency, names):
        """
        Initializes the Report buffer.

        Args:
            frequency: The writing frequency (in number of calls to write()).
            names: A list of names for the metrics to report.
        """

        self.frequency = frequency
        self.names     = names
        self.reset()

    def reset(self):
        """
        Resets the Report buffer.
        """

        self.count = 0
        self.data  = {}
        for name in self.names:
            self.data[name] = []

    def append(self, name, value):
        """
        Appends a value to the report for a given metric.

        Args:
            name: The name of the metric.
            value: The value to append.
        """

        self.data[name].append(value)

    def get(self, name):
        """
        Retrieves the data for a given metric.

        Args:
            name: The name of the metric.

        Returns:
            A list of values for the metric.
        """

        return self.data[name]

    def avg(self, name, n):
        """
        Computes the average of the last n values for a given metric.

        Args:
            name: The name of the metric.
            n: The number of last values to average.

        Returns:
            The average of the last n values.
        """

        return np.mean(self.data[name][-n:])

    def write(self, filename, force=False):
        """
        Writes the report to a file.

        Args:
            filename: The name of the file to write to.
            force: If True, forces writing regardless of the frequency.
        """

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
