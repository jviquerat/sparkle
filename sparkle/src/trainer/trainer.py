# Custom imports
from sparkle.src.core.factory      import *
from sparkle.src.trainer.regular   import *
from sparkle.src.trainer.pex_based import *

# Declare factory
trainer_factory = factory()

# Register trainers
trainer_factory.register("regular",   regular)
trainer_factory.register("pex_based", pex_based)
