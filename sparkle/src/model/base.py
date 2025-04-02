from numpy import ndarray

from sparkle.src.env.spaces import EnvSpaces


###############################################
### Base model
class BaseModel():
    def __init__(self, spaces: EnvSpaces, path: str) -> None:

        self.spaces = spaces
        self.path   = path

    @property
    def x(self) -> ndarray:
        return self.denormalize(self.x_)

    @property
    def y(self) -> ndarray:
        return self.y_

    # Normalize inputs
    def normalize(self, x: ndarray) -> ndarray:

        xx = (x - self.spaces.xmin)/(self.spaces.xmax - self.spaces.xmin)
        return xx

    # Denormalize inputs
    def denormalize(self, x: ndarray) -> ndarray:

        xx = self.spaces.xmin + (self.spaces.xmax - self.spaces.xmin)*x
        return xx
