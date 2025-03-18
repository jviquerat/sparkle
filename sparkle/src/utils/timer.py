# Generic imports
import time

# Custom imports
from sparkle.src.utils.prints import spacer, fmt_float

###############################################
### timer class
### Used to measure time spent on different operations
class timer():
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.time_tic = None
        self.time_toc = None
        self.dt_      = 0.0

    @property
    def dt(self):
        return self.dt_

    def tic(self):
        self.t_tic = time.perf_counter()

    def toc(self, show=False):
        self.t_toc = time.perf_counter()
        self.dt_  += self.t_toc - self.t_tic

        if (show): self.show()

    def show(self):
        spacer("Timer "+str(self.name)+": "+fmt_float(self.dt)+" seconds")
        self.reset()
