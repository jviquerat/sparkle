import os
import sys
import types
from typing import List, Tuple

import numpy as np

from sparkle.src.bench.bench import get_sweep_parameters, combine_parameters, combination_to_name
from sparkle.src.env.parallel import parallel
from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.pex.pex import pex_factory
from sparkle.src.plot.plot import scatter_names, violins_array
from sparkle.src.utils.json import JsonParser
from sparkle.src.utils.prints import spacer
from sparkle.src.utils.timer import Timer


class BenchPex():
    """
    Pex benchmark class
    """
    def __init__(self):

        self.name = "bench_pex"

    def run(self, args):
        """
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

        # Create paths for results and open repositories
        results_path  = 'results'
        os.makedirs(results_path, exist_ok=True)
        results_path += '/' + self.name
        os.makedirs(results_path, exist_ok=True)

        # Set parallel framework
        parallel.set({})

        # Parameters
        filename = results_path + "/" + pms.filename
        n_avg    = pms.n_avg
        sweep    = pms.sweep

        # Retrieve parameter keys and values
        keys, values = get_sweep_parameters(sweep)
        combinations = combine_parameters(keys, values)

        # Run benchmark with combinations of parameters
        # Store results in a dict mapping tuple of parameter values to numpy array
        phi_p = dict()
        minimax = dict()
        time = dict()
        for cmb in combinations:
            spacer(str(cmb))
            t, _phi_p, _minimax = self.avg(n_avg, cmb)
            phi_p[tuple(cmb.values())] = _phi_p
            minimax[tuple(cmb.values())] = _minimax
            time[tuple(cmb.values())] = t

        # Output in data file
        with open(filename, "w") as f:
            for k,v in phi_p.items():
                f.write(str(k))
                f.write("\n")
                f.write(np.array2string(v))
                f.write("\n")

        # Violin plot for phi-p
        for d in sweep.dimension:
            labels = []
            x      = []
            for m in sweep.method:
                labels += [m]
                x +=[phi_p[m, d]]

                f = results_path + "/test_"+str(d)+".png"
                t = "dimension "+str(d)
                violins_array(f, x, labels, y_label="phi_p(50)", title=t)

        # Scatter plots for given dimension
        sc_phi_p   = {}
        sc_minimax = {}
        t          = {}
        names      = []
        colors     = []
        for cmb in combinations:
            d = cmb["dimension"]
            colors.append(d)
            name = combination_to_name(cmb)
            names.append(name)
            sc_phi_p[name]   = np.mean(phi_p[tuple(cmb.values())])
            sc_minimax[name] = np.mean(minimax[tuple(cmb.values())])
            t[name]          = time[tuple(cmb.values())]

        f = os.path.join(results_path, "scatter_phip.png")
        scatter_names(f, sc_phi_p, t, names,
                      colors=colors,
                      x_label="phi_p(50)",
                      y_label="t",
                      title="scatter")

        f = os.path.join(results_path, "scatter_minimax.png")
        scatter_names(f, sc_minimax, t, names,
                      colors=colors,
                      x_label="minimax",
                      y_label="t",
                      title="scatter")

    def avg(self,
            n_avg: int,
            combination: List[dict]) -> Tuple[float, np.ndarray]:
        """
        Calculates the average phi-p metric, minimax distance and
        execution time for a given Pex method and parameters.

        Args:
            n_avg: The number of times to run the Pex algorithm for averaging.
            combination: A dictionary containing the parameters

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
        minimax = np.zeros(n_avg)
        for k in range(n_avg):
            timer_pex.tic()
            pex.reset()
            timer_pex.toc()
            phi_p[k] = pex.phi_p()
            minimax[k] = pex.minimax()

        time = timer_pex.dt/n_avg

        return time, phi_p, minimax
