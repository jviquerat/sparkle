# Custom imports
from sparkle.src.core.factory  import factory
from sparkle.src.infill.ei     import ei

# Declare factory
infill_factory = factory()

# Register kernels
infill_factory.register("ei", ei)

