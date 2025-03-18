# Generic imports
import os
import sys
import types
import itertools
import numpy as np
from collections import defaultdict

# Custom imports
from sparkle.src.env.parallel    import parallel
from sparkle.src.pex.pex         import pex_factory
from sparkle.src.env.spaces      import env_spaces
from sparkle.src.utils.distances import nearest_all_to_all
from sparkle.src.utils.timer     import timer
from sparkle.src.utils.json      import json_parser
from sparkle.src.utils.seeds     import set_seeds
from sparkle.src.utils.prints    import disclaimer, liner, spacer, bold
from sparkle.src.plot.plot       import violins_array

def avg_pex(n_avg, combination):

    dim  = combination["dimension"]
    xmin = np.zeros(dim)
    xmax = np.ones(dim)

    loc_space = {"dim": dim, "x0": None, "xmin": xmin, "xmax": xmax}
    space = env_spaces(loc_space)

    pms = types.SimpleNamespace(**combination)

    timer_pex = timer("pex")
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
    parser    = json_parser()
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
    for cmb in combinations:
        spacer(str(cmb))
        time, phi_p = avg_pex(n_avg, cmb)
        results[tuple(cmb.values())] = phi_p

    # Output in data file
    with open(filename, "w") as f:
        for k,v in results.items():
            f.write(str(k))
            f.write("\n")
            f.write(np.array2string(v))
            f.write("\n")

    # Violin plot for first set of parameters
    labels = []
    x      = []
    for m in methods:
        labels += [m]
        for d in dimensions:
            x +=[results[m, d]]

    violins_array("test.png", labels, x)

if __name__ == "__main__":
    main()
