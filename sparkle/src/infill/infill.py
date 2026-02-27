from sparkle.src.core.factory import Factory
from sparkle.src.infill.ei import EI
from sparkle.src.infill.log_ei import LogEI
from sparkle.src.infill.q_ei import QEI

# Declare factory
infill_factory = Factory()

# Register kernels
infill_factory.register("ei",     EI)
infill_factory.register("log_ei", LogEI)
infill_factory.register("q_ei",   QEI)

