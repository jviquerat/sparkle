# Custom imports
from sparkle.src.core.factory import factory
from sparkle.src.pex.random   import random
from sparkle.src.pex.lhs      import lhs

# Declare factory
pex_factory = factory()

# Register pex
pex_factory.register("random", random)
pex_factory.register("lhs",    lhs)

