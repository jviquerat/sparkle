import sys

from sparkle.tst.runner import runner


###############################################
def test_cmaes():

    # Add environment to PATH
    sys.path.append("sparkle/env/rosenbrock")

    # Run test
    runner("sparkle/tst/agent/json/rosenbrock_cmaes.json",
           4.271062449882314e-14, 5.665271017416376e-15)

###############################################
def test_cmaes_constraint():

    # Add environment to PATH
    sys.path.append("sparkle/env/constraint")

    # Run test
    runner("sparkle/tst/agent/json/constraint_cmaes.json",
           6.044096622222221e-01, 5.884923155555555e-01)
