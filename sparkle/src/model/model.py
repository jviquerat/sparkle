from sparkle.src.core.factory import Factory
from sparkle.src.model.kriging import Kriging
from sparkle.src.model.lipnet  import lipnet

# Declare factory
model_factory = Factory()

# Register models
model_factory.register("kriging", Kriging)
model_factory.register("lipnet",  lipnet)

