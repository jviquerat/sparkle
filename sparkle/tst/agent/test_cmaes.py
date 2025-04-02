import sys

from sparkle.tst.runner import runner

###############################################
### Test cmaes
def test_cmaes():

    # Add environment to PATH
    sys.path.append("sparkle/env/rosenbrock")

    # Run test
    runner("sparkle/tst/agent/json/rosenbrock_cmaes.json",
           4.271062449882314e-14, 5.665271017416376e-15)
