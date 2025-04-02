from numpy import ndarray

from sparkle.src.env.spaces import EnvSpaces


class BaseModel():
    """
    Base class for surrogate models.

    This class defines the common interface and functionality for all surrogate
    models used in the optimization framework. It provides methods for
    normalizing and denormalizing input data.
    """
    def __init__(self, spaces: EnvSpaces, path: str) -> None:
        """
        Initializes the BaseModel.

        Args:
            spaces: The environment's search space definition.
            path: The base path for storing results.
        """

        self.spaces = spaces
        self.path   = path

    @property
    def x(self) -> ndarray:
        """
        Returns the denormalized input data points.
        """
        return self.denormalize(self.x_)

    @property
    def y(self) -> ndarray:
        """
        Returns the target values.
        """
        return self.y_

    def normalize(self, x: ndarray) -> ndarray:
        """
        Normalizes input data points to the range [0, 1].

        Args:
            x: The input data points to normalize.

        Returns:
            The normalized data points.
        """

        xx = (x - self.spaces.xmin)/(self.spaces.xmax - self.spaces.xmin)
        return xx

    def denormalize(self, x: ndarray) -> ndarray:
        """
        Denormalizes input data points from the range [0, 1] to the original scale.

        Args:
            x: The input data points to denormalize.

        Returns:
            The denormalized data points.
        """

        xx = self.spaces.xmin + (self.spaces.xmax - self.spaces.xmin)*x
        return xx
