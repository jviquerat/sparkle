from sparkle.src.plot.plot import plot_avg
from sparkle.src.utils.data import DataAvg


def average(args):
    """
    Averages data from multiple runs and generates a plot.

    This function takes a list of data files, averages the data across them,
    writes the averaged data to a new file, and generates a plot of the
    averaged data.

    Args:
        args: A list of file paths to data files to be averaged.
    """

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
