import time

from sparkle.src.utils.prints import fmt_float, spacer


###############################################
### timer class
### Used to measure time spent on different operations
class Timer():
    """
    Timer class.

    This class provides methods for measuring the time spent on different
    operations.
    """
    def __init__(self, name: str) -> None:
        """
        Initializes the Timer.

        Args:
            name: The name of the timer.
        """
        self.name = name
        self.reset()

    def reset(self) -> None:
        """
        Resets the timer.
        """
        self.time_tic = None
        self.time_toc = None
        self.dt_      = 0.0

    @property
    def dt(self) -> float:
        """
        Returns the elapsed time.
        """
        return self.dt_

    def tic(self) -> None:
        """
        Starts the timer.
        """
        self.t_tic = time.perf_counter()

    def toc(self, show: bool=False) -> None:
        """
        Stops the timer and optionally displays the elapsed time.

        Args:
            show: If True, displays the elapsed time.
        """
        self.t_toc = time.perf_counter()
        self.dt_  += self.t_toc - self.t_tic

        if (show): self.show()

    def show(self) -> None:
        """
        Displays the elapsed time.
        """
        spacer("Timer "+str(self.name)+": "+fmt_float(self.dt)+" seconds")
        self.reset()
