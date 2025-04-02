from sparkle.src.plot.plot import plot_avg
from sparkle.src.utils.data import DataAvg


# Average existing runs
def average(args):

    # Count arguments
    n_args = len(args)

    # Get length of file
    with open(args[0], 'r') as f:
        n_lines = sum(1 for line in f)

    # Intialize averager
    averager = DataAvg(2, n_lines, n_args)

    # Run
    for run in range(n_args):
        filename = args[run]
        averager.store(filename, run)

    # Write to file
    filename = 'avg.dat'
    data = averager.average(filename)

    # Plot
    filename = 'avg'
    plot_avg(data, filename, "linear")
