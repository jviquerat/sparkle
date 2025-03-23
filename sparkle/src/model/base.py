# Generic imports

###############################################
### Base model
class base_model():
    def __init__(self, spaces, path):

        self.spaces = spaces
        self.path   = path

    @property
    def x(self):
        return self.denormalize(self.x_)

    @property
    def y(self):
        return self.y_

    # Normalize inputs
    def normalize(self, x):

        xx = (x - self.spaces.xmin)/(self.spaces.xmax - self.spaces.xmin)
        return xx

    # Denormalize inputs
    def denormalize(self, x):

        xx = self.spaces.xmin + (self.spaces.xmax - self.spaces.xmin)*x
        return xx
