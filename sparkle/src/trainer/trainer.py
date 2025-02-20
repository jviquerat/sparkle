# Custom imports
from sparkle.src.core.factory      import factory
from sparkle.src.trainer.regular   import regular
from sparkle.src.trainer.metamodel import metamodel

# Declare factory
trainer_factory = factory()

# Register trainers
trainer_factory.register("regular",   regular)
trainer_factory.register("metamodel", metamodel)
