import numpy as np

from sparkle.env.base_env import base_env
from sparkle.src.utils.default import set_default
from sparkle.src.utils.error import error


###############################################
class constraint(base_env):
    """
    Parabola function in dimension 2 with parameter constraints:

    min  x² + y²
    s.t. 1 - x³ - y < 0

    The constraints can be handled in two ways:
        - a priori, using constraint_type = "a_priori"
          in this case, sample validity will be checked a priori
        - a posteriori, using constraint_type = "a_posteriori"
          in this case, a penalty term is added to the cost
    """
    def __init__(self, cpu: int, path: str, pms=None):

        # Fill structure
        self.name      = 'constraint'
        self.base_path = path
        self.cpu       = cpu
        self.dim       = 2

        self.x0        = set_default("x0",   4.0*np.ones(self.dim), pms)
        self.xmin      = set_default("xmin",-5.0*np.ones(self.dim), pms)
        self.xmax      = set_default("xmax", 5.0*np.ones(self.dim), pms)

        self.constraint_type = set_default("constraint_type", "a_priori", pms)

        if self.constraint_type not in ["a_priori", "a_posteriori"]:
            error("constraint", "__init__",
                  "Invalid constraint type, valid types are a_priori and a_posteriori")

        # Plotting data
        self.it_plt    = 0
        self.vmin      = 0.0
        self.vmax      = 20.0
        self.levels    = [0.1, 1.0, 5.0, 10.0, 20.0]

    def reset(self, run: int) -> bool:

        self.path   = self.base_path+"/"+str(run)
        self.it_plt = 0

        return True

    def validate(self, x: np.ndarray) -> bool:
        """
        Sample validation function for the constraint
        If the constraint is "a_posteriori", all samples are valid and the
        constraint is handled with a penalty term in the cost function

        Args:
            x: numpy array with a single sample point

        Returns:
            a boolean to indicate the validity of the sample
        """
        if self.constraint_type == "a_priori":
            return np.where(x[0]**3 + x[1] > 1.0, True, False)
        else:
            return True

    def cost(self, x: np.ndarray) -> float:
        """
        Cost function with possible constraint
        - If the constraint is "a_priori", the direct cost is returned (NaN is returned for
          invalid samples when the function is called for plotting purpose)
        - If the constraint is "a_posteriori", a penalty is added to the cost function:
              min  x² + y²
              s.t. 1 - x³ - y < 0
          is transformed into:
              min x² + y² + max(0, K*h(x,y))
              with h(x,y) = 1 - y - x³
              and K a large number
        """
        if self.constraint_type == "a_priori":
            if self.validate(x):
                return x[0]**2 + x[1]**2
            else:
                return np.nan
        else:
            K = 100.0
            hx = 1.0 - x[1] - x[0]**3
            return x[0]**2 + x[1]**2 + K*max(0.0, hx)

    def close(self):
        pass
