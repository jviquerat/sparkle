# Custom imports
from sparkle.src.core.factory  import factory
from sparkle.src.model.kriging import kriging

# Declare factory
model_factory = factory()

# Register models
model_factory.register("kriging", kriging)

