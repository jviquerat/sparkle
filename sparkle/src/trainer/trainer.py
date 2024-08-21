# Custom imports
from sparkle.src.core.factory      import factory
from sparkle.src.trainer.regular   import regular
from sparkle.src.trainer.pex_based import pex_based

# Declare factory
trainer_factory = factory()

# Register trainers
trainer_factory.register("regular",   regular)
trainer_factory.register("pex_based", pex_based)
