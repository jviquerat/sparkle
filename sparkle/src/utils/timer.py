import time

from sparkle.src.utils.prints import fmt_float, spacer


###############################################
### timer class
### Used to measure time spent on different operations
class Timer():
    def __init__(self, name: str) -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.time_tic = None
        self.time_toc = None
        self.dt_      = 0.0

    @property
    def dt(self) -> float:
        return self.dt_

    def tic(self) -> None:
        self.t_tic = time.perf_counter()

    def toc(self, show: bool=False) -> None:
        self.t_toc = time.perf_counter()
        self.dt_  += self.t_toc - self.t_tic

        if (show): self.show()

    def show(self) -> None:
        spacer("Timer "+str(self.name)+": "+fmt_float(self.dt)+" seconds")
        self.reset()
