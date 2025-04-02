from sparkle.src.core.factory import Factory
from sparkle.src.agent.pso    import PSO
from sparkle.src.agent.cmaes  import CMAES
from sparkle.src.agent.cem    import CEM
from sparkle.src.agent.pbo    import PBO
from sparkle.src.agent.ego    import EGO

# Declare factory
agent_factory = Factory()

# Register agents
agent_factory.register("pso",   PSO)
agent_factory.register("cmaes", CMAES)
agent_factory.register("cem",   CEM)
agent_factory.register("pbo",   PBO)
agent_factory.register("ego",   EGO)
