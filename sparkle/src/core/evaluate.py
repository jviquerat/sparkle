import os
import sys

import numpy as np

from sparkle.src.env.parallel import parallel
from sparkle.src.utils.json import JsonParser


def evaluate(dat_file, json_file):
    """
    Evaluates the best sample found during optimization.

    This function loads the best sample from a data file, sets up the
    environment based on a JSON configuration file, evaluates the cost
    of the best sample in the environment, and renders the result.

    Args:
        dat_file: The path to the data file containing the best sample.
        json_file: The path to the JSON configuration file.
    """

    # Add paths to PATH
    base_path  = os.path.abspath(os.getcwd())
    dat_path   = os.path.dirname(dat_file)
    json_path  = os.path.dirname(json_file)
    sys.path.append(base_path)
    sys.path.append(dat_path)
    sys.path.append(json_path)

    # Initialize environment
    parser = JsonParser()
    pms    = parser.read(json_file)
    parallel.set(pms)
    env    = parallel.environments(".", pms.trainer.environment)
    env.reset(0)

    # Retrieve data file
    data = np.loadtxt(dat_file)
    c = data[-1,2] # best cost
    s = data[-1,3] # best step
    x = data[int(s),4:] # best x

    # Run environment
    x = np.array([x])
    c = env.cost(x)
    env.render(x, c)

    # Finalize
    env.close()
    parallel.finalize()
