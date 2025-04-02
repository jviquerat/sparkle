import itertools
import sys
import types
from collections import defaultdict

import numpy as np

from sparkle.src.env.parallel import parallel
from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.pex.pex import pex_factory
from sparkle.src.plot.plot import scatter_names, violins_array
from sparkle.src.utils.json import JsonParser
from sparkle.src.utils.prints import bold, disclaimer, liner, spacer
from sparkle.src.utils.timer import Timer


def AvgPex(n_avg, combination):

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
    # "method" and "dimension" are mandatory keys
    keys   = ["method", "dimension"]
    values = [ methods,  dimensions]

    # Check inner parameters (passed to pex through pms)
    if (hasattr(pms, "inner_pms")):
        iterable = pms.inner_pms.__dict__.items()
        for k in iterable:
            keys   += [k[0]]
            values += [k[1]]

    spacer("Parameter keys: "+str(keys))
    spacer("Parameter values: "+str(values))

    # Generate combinations as list of tuples
    comb_tuples = list(itertools.product(*values))

    # Convert list of tuples to list of dicts
    combinations = []
    for k in comb_tuples:
        comb_dict = defaultdict(list)
        for l in range(len(k)):
            comb_dict[keys[l]] = k[l]
        combinations.append(dict(comb_dict))
    spacer("Nb of combinations: "+str(len(combinations)))

    # Run benchmark with combinations of parameters
    # Store results in a dict mapping tuple of parameter values to numpy array
    results = dict()
    time    = dict()
    for cmb in combinations:
        spacer(str(cmb))
        t, phi_p = AvgPex(n_avg, cmb)
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
    phi_p = {}
    t     = {}
    d     = 10
    for m in methods:
        phi_p[m] = np.mean(results[m,d])
        t[m]     = time[m,d]

    f = "scatter.png"
    scatter_names(f, phi_p, t, methods, x_label="phi_p", y_label="t", title="scatter")


if __name__ == "__main__":
    main()
