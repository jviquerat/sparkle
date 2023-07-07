# Custom imports
from sparkle.src.core.factory import *
from sparkle.src.agent.pso    import *

# Declare factory
agent_factory = factory()

# Register agents
agent_factory.register("pso", pso)
