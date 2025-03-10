from sparkle.src.core.factory import Factory
from sparkle.src.model.kriging import Kriging
from sparkle.src.model.lipnet import LipNet
from sparkle.src.model.sepnet  import sepnet

# Declare factory
model_factory = Factory()

# Register models
model_factory.register("kriging", Kriging)
model_factory.register("lipnet",  LipNet)
model_factory.register("sepnet",  sepnet)

