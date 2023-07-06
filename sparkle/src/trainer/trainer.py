# Custom imports
from sparkle.src.core.factory    import *
from sparkle.src.trainer.regular import *

# Declare factory
trainer_factory = factory()

# Register trainers
trainer_factory.register("regular", regular)
