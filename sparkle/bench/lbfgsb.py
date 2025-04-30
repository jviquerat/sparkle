import os
import sys
import types
from typing import List, Tuple

import numpy as np

from sparkle.src.bench.bench import (get_sweep_parameters,
                                     combine_parameters,
                                     combination_to_name,
                                     write_bench_data)
from sparkle.src.env.parallel import parallel
from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.agent.ms_lbfgsb import MSLBFGSB
from sparkle.env.sinebump.sinebump import sinebump
from sparkle.src.utils.json import JsonParser
from sparkle.src.utils.prints import spacer
from sparkle.src.utils.timer import Timer
from sparkle.src.plot.plot import scatter_names


class BenchLBFGSB():
    """
    L-BFGS-B benchmark class
    """
    def __init__(self):

        self.name = "bench_lbfgsb"

    def run(self, args):
        """
        Runs the L-BFGS-B benchmark.
        Reads parameters, generates combinations, runs the benchmark,
        and outputs results and plots.
        """

        # Initialize json parser and read parameters
        json_file = args[args.index("-json")+1]
        parser    = JsonParser()
        pms       = parser.read(json_file)

        # Create paths for results
        results_path  = 'results'
        os.makedirs(results_path, exist_ok=True)
        results_path = os.path.join(results_path, self.name)
        os.makedirs(results_path, exist_ok=True)

        # Set parallel framework
        parallel.set({})

        # Parameters from JSON
        filename = os.path.join(results_path, pms.filename)
        n_avg    = pms.n_avg
        sweep    = pms.sweep

        # Retrieve parameter keys and values for combinations
        keys, values = get_sweep_parameters(sweep)
        combinations = combine_parameters(keys, values)

        # Run benchmark for each combination
        cost = dict()
        time = dict()
        for cmb in combinations:
            spacer(str(cmb))
            t, costs = self.avg(n_avg, cmb)
            cost[tuple(cmb.values())] = costs
            time[tuple(cmb.values())] = t

        # Output in data file
        write_bench_data(filename, cost)

        # Generate Plots
        avg_costs = {}
        avg_times = {}
        names = []
        colors = []

        for cmb in combinations:
            colors.append(cmb["m"])
            name = combination_to_name(cmb)
            names.append(name)
            avg_costs[name] = np.mean(cost[tuple(cmb.values())])
            avg_times[name] = time[tuple(cmb.values())]

        f = os.path.join(results_path, "scatter_time_vs_cost.png")
        scatter_names(f, avg_costs, avg_times, names, colors=colors,
                      x_label="Average Final Cost (log scale)",
                      y_label="Average Execution Time (s, log scale)",
                      title="Avg Time vs. Avg Cost")

    def avg(self,
            n_avg: int,
            combination: List[dict]) -> Tuple[float, np.ndarray]:
        """
        Calculates the average final cost and execution time for L-BFGS-B
        with given parameters on the Sinebump function.

        Args:
            n_avg: Number of averaging runs.
            combination: A dictionary containing the parameters

        Returns:
            A tuple containing:
                - Average execution time (float).
                - NumPy array of final costs for each run.
        """
        env = sinebump(cpu=0, path='.')
        xmin = env.xmin
        xmax = env.xmax
        cost_func = env.cost

        opt = MSLBFGSB()
        timer_lbfgsb = Timer("lbfgsb_run")
        final_costs = np.zeros(n_avg)

        for k in range(n_avg):
            timer_lbfgsb.tic()
            # We use the Multi-Start version for better chance of finding global optimum
            x_opt, c_opt = opt.optimize(cost_func,
                                        xmin,
                                        xmax,
                                        df=None, # Use finite differences for gradient
                                        n_pts=combination["n_pts_ms"],
                                        m=combination["m"],
                                        tol=combination["tol"],
                                        max_iter=combination["max_iter"])
            timer_lbfgsb.toc()
            final_costs[k] = c_opt

        avg_time = timer_lbfgsb.dt / n_avg

        return avg_time, final_costs

