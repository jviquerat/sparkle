# Custom imports
from sparkle.src.core.factory import *
from sparkle.src.agent.pso    import *
from sparkle.src.agent.cmaes  import *
from sparkle.src.agent.cem    import *
from sparkle.src.agent.pbo    import *

# Declare factory
agent_factory = factory()

# Register agents
agent_factory.register("pso",   pso)
agent_factory.register("cmaes", cmaes)
agent_factory.register("cem",   cem)
agent_factory.register("pbo",   pbo)
