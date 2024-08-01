# Custom imports
from sparkle.src.core.factory import factory
from sparkle.src.agent.pso    import pso
from sparkle.src.agent.cmaes  import cmaes
from sparkle.src.agent.cem    import cem
from sparkle.src.agent.pbo    import pbo
from sparkle.src.agent.ego    import ego

# Declare factory
agent_factory = factory()

# Register agents
agent_factory.register("pso",   pso)
agent_factory.register("cmaes", cmaes)
agent_factory.register("cem",   cem)
agent_factory.register("pbo",   pbo)
agent_factory.register("ego",   ego)
