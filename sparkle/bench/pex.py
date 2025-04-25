import sys
import types
from typing import List, Tuple

import numpy as np

from sparkle.src.bench.bench import combine_parameters
from sparkle.src.env.parallel import parallel
from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.pex.pex import pex_factory
from sparkle.src.plot.plot import scatter_names, violins_array
from sparkle.src.utils.json import JsonParser
from sparkle.src.utils.prints import bold, disclaimer, liner, spacer
from sparkle.src.utils.timer import Timer


def avg_pex(n_avg: int, combination: List[dict]) -> Tuple[float, np.ndarray]:
    """
    Calculates the average phi-p metric and execution time for a given
    Pex method and parameters.

    Args:
        n_avg: The number of times to run the Pex algorithm for averaging.
        combination: A dictionary containing the parameters for the Pex algorithm,
                     including 'dimension' and 'method'.

    Returns:
        A tuple containing:
            - The average execution time (float).
            - A NumPy array of phi-p metric values for each run.
    """

    dim  = combination["dimension"]
    xmin = np.zeros(dim)
    xmax = np.ones(dim)

    loc_space = {"dim": dim, "x0": None, "xmin": xmin, "xmax": xmax}
    space = EnvSpaces(loc_space)

    pms = types.SimpleNamespace(**combination)

    timer_pex = Timer("pex")
    pex = pex_factory.create(combination["method"],
                             spaces = space,
                             pms    = pms)

    phi_p = np.zeros(n_avg)
    for k in range(n_avg):
        timer_pex.tic()
        pex.reset()
        timer_pex.toc()
        phi_p[k] = pex.phi_p()

    time = timer_pex.dt/n_avg

    return time, phi_p

def main():
    """
    Main function to run the Pex benchmark.

    This function reads parameters from a JSON file, generates combinations of
    parameters, runs the benchmark for each combination, and outputs the results
    to a data file and plots.
    """

    # Check arguments
    args = sys.argv

    # Initialize json parser and read parameters
    json_file = args[args.index("-json")+1]
    parser    = JsonParser()
    pms       = parser.read(json_file)

    # Set parallel framework
    parallel.set({})

    # Printings
    disclaimer()
    liner(bold('Pex benchmark'))

    # Parameters
    filename   = pms.filename
    n_avg      = pms.n_avg
    methods    = pms.methods
    dimensions = pms.dimensions

    # Retrieve parameter keys and values
    keys   = ["method", "dimension"]
    values = [ methods,  dimensions]

    combinations = combine_parameters(keys, values)

    # Run benchmark with combinations of parameters
    # Store results in a dict mapping tuple of parameter values to numpy array
    results = dict()
    time    = dict()
    for cmb in combinations:
        spacer(str(cmb))
        t, phi_p = avg_pex(n_avg, cmb)
        results[tuple(cmb.values())] = phi_p
        time[tuple(cmb.values())]    = t

    # Output in data file
    with open(filename, "w") as f:
        for k,v in results.items():
            f.write(str(k))
            f.write("\n")
            f.write(np.array2string(v))
            f.write("\n")

    # Violin plot for phi-p
    for d in dimensions:
        labels = []
        x      = []
        for m in methods:
            labels += [m]
            x +=[results[m, d]]

            f = "test_"+str(d)+".png"
            t = "dimension "+str(d)
            violins_array(f, x, labels, y_label="phi_p(50)", title=t)

    # Scatter plots for given dimension
    phi_p  = {}
    t      = {}
    names  = []
    colors = []
    for cmb in combinations:
        d = cmb["dimension"]
        m = cmb["method"]
        colors.append(d)
        name = f"{m} {d}"
        names.append(name)
        phi_p[name] = 1.0/np.mean(results[m,d])
        t[name]     = time[m,d]

    f = "scatter.png"
    scatter_names(f, phi_p, t, names, colors=colors, x_label="1/phi_p(50)", y_label="t", title="scatter")


if __name__ == "__main__":
    main()
