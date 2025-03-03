# Custom imports
from sparkle.src.core.factory  import factory
from sparkle.src.infill.ei     import ei
from sparkle.src.infill.log_ei import log_ei

# Declare factory
infill_factory = factory()

# Register kernels
infill_factory.register("ei",     ei)
infill_factory.register("log_ei", log_ei)

