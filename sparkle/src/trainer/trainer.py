from sparkle.src.core.factory import Factory
from sparkle.src.trainer.metamodel import Metamodel
from sparkle.src.trainer.regular import Regular
from sparkle.src.trainer.separable import Separable

# Declare factory
trainer_factory = Factory()

# Register trainers
trainer_factory.register("regular",   Regular)
trainer_factory.register("metamodel", Metamodel)
trainer_factory.register("separable", Separable)

