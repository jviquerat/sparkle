import os
import sys

import numpy as np

from sparkle.src.env.parallel import parallel
from sparkle.src.utils.json import JsonParser


# Evaluate best sample
def evaluate(dat_file, json_file):

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
    env    = parallel.environments(".", pms.environment)
    env.reset(0)

    # Retrieve data file
    data = np.loadtxt(dat_file)
    c = data[-1,2] # best cost
    s = data[-1,3] # best step
    x = data[int(s),4:] # best x

    # Run environment
    x = np.array([x])
    c = env.cost(x)
    env.render(x)

    # Finalize
    env.close()
    parallel.finalize()
