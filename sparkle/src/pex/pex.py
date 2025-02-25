# Custom imports
from sparkle.src.core.factory           import factory
from sparkle.src.pex.random             import random
from sparkle.src.pex.lhs                import lhs
from sparkle.src.pex.maximin_lhs        import maximin_lhs
from sparkle.src.pex.fixed_poisson_disc import fixed_poisson_disc

# Declare factory
pex_factory = factory()

# Register pex
pex_factory.register("random",             random)
pex_factory.register("lhs",                lhs)
pex_factory.register("maximin_lhs",        maximin_lhs)
pex_factory.register("fixed_poisson_disc", fixed_poisson_disc)

