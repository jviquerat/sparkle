# Custom imports
from sparkle.src.core.factory    import *
from sparkle.src.optimizer.adam  import *
from sparkle.src.optimizer.adamw import *
from sparkle.src.optimizer.lbfgs import *

# Declare factory
opt_factory = factory()

# Register agents
opt_factory.register("adam",  adam)
opt_factory.register("adamw", adamw)
opt_factory.register("lbfgs", lbfgs)
