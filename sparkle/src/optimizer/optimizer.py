from sparkle.src.core.factory    import Factory
from sparkle.src.optimizer.adam  import Adam
from sparkle.src.optimizer.adamw import AdamW
from sparkle.src.optimizer.lbfgs import LBFGS

# Declare factory
opt_factory = Factory()

# Register agents
opt_factory.register("adam",  Adam)
opt_factory.register("adamw", AdamW)
opt_factory.register("lbfgs", LBFGS)
