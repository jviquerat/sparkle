from sparkle.src.core.factory  import Factory
from sparkle.src.model.kriging import Kriging

# Declare factory
model_factory = Factory()

# Register models
model_factory.register("kriging", Kriging)

