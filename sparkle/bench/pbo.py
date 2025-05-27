import os
import sys
import types
from typing import List, Tuple, Any

import numpy as np

from sparkle.src.bench.bench import (get_sweep_parameters,
                                     combine_parameters,
                                     combination_to_name)
from sparkle.src.env.parallel import parallel
from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.plot.plot import scatter_names
from sparkle.src.utils.json import JsonParser
from sparkle.src.utils.prints import spacer
from sparkle.src.utils.timer import Timer
from sparkle.src.trainer.trainer import trainer_factory
from sparkle.src.utils.default import set_default


class BenchPBO():
    """
    Pex benchmark class
    """
    def __init__(self):

        self.name = "bench_pbo"

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

        # Add hint path to PATH
        base_path = os.path.abspath(os.getcwd())
        hint_path = pms.hint_path
        hint_path = os.path.join(base_path, hint_path)
        sys.path.append(hint_path)

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
        color_by = set_default("color_by", None, pms)

        # Retrieve parameter keys and values
        keys, values = get_sweep_parameters(sweep)
        combinations = combine_parameters(keys, values)

        # Run benchmark with combinations of parameters
        # Store results in a dict mapping tuple of parameter values to numpy array
        best = dict()
        time = dict()
        for cmb in combinations:
            spacer(str(cmb))
            b = self.avg(n_avg, cmb, pms, results_path)
            best[tuple(cmb.values())] = b

        # Scatter plots for given dimension
        best_mean = {}
        best_std  = {}
        names     = []
        colors    = []
        if color_by is None:
            color_by = list(cmb)[0]

        for cmb in combinations:
            d = cmb[color_by]
            colors.append(d)
            name = combination_to_name(cmb)
            names.append(name)
            best_mean[name] = np.mean(best[tuple(cmb.values())])
            best_std[name]  = np.std(best[tuple(cmb.values())])

        f = os.path.join(results_path, "best_score.png")
        scatter_names(f, best_mean, best_std, names,
                      colors=colors,
                      use_x_log_scale=True,
                      use_y_log_scale=True,
                      x_label="best mean score",
                      y_label="best std score",
                      title=", ".join(keys))

    def avg(self,
            n_avg: int,
            combination: List[dict],
            pms: Any,
            results_path: str) -> Tuple[float, np.ndarray]:
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
        pms.trainer.agent.silent = True
        for k,v in combination.items():
            pms.trainer.agent.k = v
        trainer = trainer_factory.create(pms.trainer.name,
                                         path      = results_path,
                                         pms       = pms.trainer)

        best = np.zeros(n_avg)
        for k in range(n_avg):
            os.makedirs(results_path+'/'+str(k), exist_ok=True)
            trainer.reset(k)
            trainer.optimize()
            best[k] = trainer.hist_b[-1]

        return best
