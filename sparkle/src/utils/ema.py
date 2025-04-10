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
