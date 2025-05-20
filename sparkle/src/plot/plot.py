import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
from numpy import ndarray

from adjustText import adjust_text

from sparkle.src.env.spaces import EnvSpaces
from sparkle.src.utils.error import error

plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['figure.titleweight'] = 'bold'

def plot_avg(data: ndarray, filename: str, avg_type: str) -> None:
    """
    Plots the average and best cost over optimization steps.

    This function generates a plot showing the average and best cost
    values over the course of optimization steps, along with their
    respective standard deviations.

    Args:
        data: A NumPy array containing the data to plot. The array should
            have columns for step, average cost, average cost plus std,
            average cost minus std, best cost, best cost plus std, and
            best cost minus std.
        filename: The base filename for saving the plot.
        avg_type: The type of scaling to use for the y-axis ('linear' or 'log').
    """

    stp   = data[:,0]

    # Average cost
    avg   = data[:,1]
    p     = data[:,2]
    m     = data[:,3]

    avg_avg, avg_p, avg_m, ylabel = return_plottables(avg, p, m, avg_type)

    # Best cost
    avg   = data[:,4]
    p     = data[:,5]
    m     = data[:,6]

    bst_avg, bst_p, bst_m, ylabel = return_plottables(avg, p, m, avg_type)

    # Actual plotting
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    fig.suptitle(filename)

    ax.set_xlabel('evaluations')
    ax.set_ylabel(ylabel)
    plt.yscale(avg_type)
    plt.plot(stp, avg_avg, color='blue', label='avg cost')
    plt.plot(stp, bst_avg, color='red',  label='bst cost')
    plt.fill_between(stp, avg_p, avg_m, alpha=0.4, color='blue')
    plt.fill_between(stp, bst_p, bst_m, alpha=0.4, color='red')
    plt.grid(True)
    plt.legend()

    fig.tight_layout()
    fig.savefig(filename+'.png')

def return_plottables(avg: ndarray,
                      p: ndarray,
                      m: ndarray,
                      avg_type: str) -> tuple[ndarray, ndarray, ndarray, str]:
    """
    Returns the plottable data based on the specified average type.

    This function prepares the data for plotting by applying the
    appropriate scaling (linear or logarithmic) to the average,
    plus standard deviation, and minus standard deviation values.

    Args:
        avg: A NumPy array of average values.
        p: A NumPy array of average plus standard deviation values.
        m: A NumPy array of average minus standard deviation values.
        avg_type: The type of scaling to use ('linear' or 'log').

    Returns:
        A tuple containing:
            - The (potentially transformed) average values.
            - The (potentially transformed) plus standard deviation values.
            - The (potentially transformed) minus standard deviation values.
            - The label for the y-axis.

    Raises:
        ValueError: If the avg_type is not 'linear' or 'log'.
    """

    if (avg_type not in ["linear", "log"]):
        error("plot", "return_plottable", "avg_type should be either linear or log")

    if (avg_type == "linear"):
        ylabel = "cost"

    if (avg_type == "log"):
        log_avg = np.log(avg)
        log_std = 0.434*(p-avg)/avg
        log_p   = log_avg + log_std
        log_m   = log_avg - log_std
        p       = np.exp(log_p)
        m       = np.exp(log_m)
        ylabel  = "log(cost)"

    return avg, p, m, ylabel

def render_1D_regular(filename: str,
                      spaces: EnvSpaces,
                      x: ndarray,
                      c: ndarray,
                      x_plot: ndarray,
                      cost_map: ndarray) -> None:
    """
    Renders a 1D plot for the regular trainer.

    This function generates a 1D plot showing the cost function and the
    sampled points.

    Args:
        filename: The filename for saving the plot.
        spaces: The environment's search space definition.
        x: A NumPy array of sampled points.
        c: A NumPy array of cost values for the sampled points.
        x_plot: A NumPy array of x-coordinates for plotting the cost function.
        cost_map: A NumPy array of cost values for plotting the cost function.
    """

    plt.clf()
    fig = plt.figure()
    ax  = fig.add_subplot(111)

    ax.set_xlim([spaces.xmin[0], spaces.xmax[0]])
    ax.set_ylim([spaces.vmin,    spaces.vmax])
    ax.grid()
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.plot(x_plot, cost_map, label="f(x)")
    ax.set_ylabel('y')

    ax.scatter(x[:,0], c[:], c="black", marker='o', alpha=0.8, label="samples")
    ax.legend(loc='upper left')

    fig.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()

def render_2D_regular(filename: str,
                      spaces: EnvSpaces,
                      x: ndarray,
                      c: ndarray,
                      x_plot: ndarray,
                      y_plot: ndarray,
                      cost_map: ndarray) -> None:
    """
    Renders a 2D plot for the regular trainer.

    This function generates a 2D plot showing the cost function as a
    contour plot and the sampled points as scatter points.

    Args:
        filename: The filename for saving the plot.
        spaces: The environment's search space definition.
        x: A NumPy array of sampled points.
        c: A NumPy array of cost values for the sampled points.
        x_plot: A NumPy array of x-coordinates for plotting the cost function.
        y_plot: A NumPy array of y-coordinates for plotting the cost function.
        cost_map: A NumPy array of cost values for plotting the cost function.
    """

    plt.clf()
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    fig.set_size_inches(3, 3)
    fig.subplots_adjust(0,0,1,1)

    cmap = matplotlib.colormaps.get_cmap('RdBu_r')
    cmap.set_bad(color='black')

    ax.set_xlim([spaces.xmin[0], spaces.xmax[0]])
    ax.set_ylim([spaces.xmin[1], spaces.xmax[1]])
    ax.axis('off')
    ax.imshow(cost_map,
              extent=[spaces.xmin[0], spaces.xmax[0],
                      spaces.xmin[1], spaces.xmax[1]],
              vmin=spaces.vmin, vmax=spaces.vmax,
              alpha=0.8, cmap=cmap)

    cnt = ax.contour(x_plot, y_plot, cost_map, levels=spaces.levels,
                     colors='black', alpha=0.5)
    ax.clabel(cnt, inline=True, fontsize=8, fmt="%.0f")
    ax.scatter(x[:,0], x[:,1], c="black", marker='o', alpha=0.8)

    plt.savefig(filename, dpi=100)
    plt.close()

def render_1D_metamodel(filename: str,
                        spaces: EnvSpaces,
                        x: ndarray,
                        c: ndarray,
                        x_plot: ndarray,
                        cost_map: ndarray,
                        y_mu: ndarray,
                        y_std: ndarray,
                        fct: ndarray,
                        fct_name: str,
                        highlight_last: bool=True) -> None:
    """
    Renders a 1D plot for the metamodel trainer.

    This function generates a 1D plot showing the cost function, the
    sampled points, the metamodel's prediction, and the confidence
    interval.

    Args:
        filename: The filename for saving the plot.
        spaces: The environment's search space definition.
        x: A NumPy array of sampled points.
        c: A NumPy array of cost values for the sampled points.
        x_plot: A NumPy array of x-coordinates for plotting.
        cost_map: A NumPy array of cost values for plotting the cost function.
        y_mu: A NumPy array of the metamodel's mean predictions.
        y_std: A NumPy array of the metamodel's standard deviation predictions.
        fct: A NumPy array of values for an additional function to plot.
        fct_name: The name of the additional function.
        highlight_last: Whether to highlight the last sampled point.
    """

    plt.clf()
    fig = plt.figure()

    ax = fig.add_subplot(211)
    ax.set_xlim([spaces.xmin[0], spaces.xmax[0]])
    ax.set_ylim([spaces.vmin,    spaces.vmax])
    ax.grid()
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.plot(x_plot, cost_map, label="f(x)", zorder=0)
    ax.set_ylabel('y')

    ax.scatter(x[:,0], c[:], c="black", marker='o', alpha=0.5, label="samples", zorder=1)
    ax.legend(loc='upper left')

    ax.plot(x_plot, y_mu, linestyle='dashed', label="model", zorder=0)
    ax.fill_between(x_plot, y_mu-y_std, y_mu+y_std, alpha=0.2,
                    label="confidence interval", zorder=0)
    if highlight_last:
        ax.scatter(x[-1,0], c[-1], c='red', marker='o', alpha=0.5, zorder=1)

    ax = fig.add_subplot(212)
    ax.set_xlim([spaces.xmin[0], spaces.xmax[0]])
    ratio = np.max(abs(fct))/np.min(abs(fct) + 1.0e-5)
    if ratio > 100:
        if np.any(fct < 0.0):
            ax.set_yscale("symlog")
        else:
            ax.set_yscale("log")
    ax.plot(x_plot, fct, color='r')
    ax.grid()
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_ylabel(fct_name)

    fig.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()

def render_2D_metamodel(filename: str,
                        spaces: EnvSpaces,
                        x: ndarray,
                        c: ndarray,
                        x_plot: ndarray,
                        y_plot: ndarray,
                        cost_map: ndarray,
                        y_mu: ndarray,
                        y_std: ndarray,
                        fct: ndarray,
                        fct_name: str,
                        highlight_last: bool=True) -> None:
    """
    Renders a 2D plot for the metamodel trainer.

    This function generates a 2D plot showing the cost function, the
    metamodel's prediction, and an additional function, along with the
    sampled points.

    Args:
        filename: The filename for saving the plot.
        spaces: The environment's search space definition.
        x: A NumPy array of sampled points.
        c: A NumPy array of cost values for the sampled points.
        x_plot: A NumPy array of x-coordinates for plotting.
        y_plot: A NumPy array of y-coordinates for plotting.
        cost_map: A NumPy array of cost values for plotting the cost function.
        y_mu: A NumPy array of the metamodel's mean predictions.
        y_std: A NumPy array of the metamodel's standard deviation predictions.
        fct: A NumPy array of values for an additional function to plot.
        fct_name: The name of the additional function.
        highlight_last: Whether to highlight the last sampled point.
    """

    plt.clf()
    fig = plt.figure()
    fig.set_size_inches(8, 3)
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.01, top=0.99, left=0.01, right=0.99)
    plt.rcParams.update({'axes.titlesize': 'small'})

    ax = fig.add_subplot(131)
    ax.set_xlim([spaces.xmin[0], spaces.xmax[0]])
    ax.set_ylim([spaces.xmin[1], spaces.xmax[1]])
    ax.axis('off')
    im1 = ax.imshow(cost_map,
                    extent=[spaces.xmin[0], spaces.xmax[0],
                            spaces.xmin[1], spaces.xmax[1]],
                    vmin=spaces.vmin, vmax=spaces.vmax, alpha=0.8, cmap='RdBu_r')

    cnt = ax.contour(x_plot, y_plot, cost_map, levels=spaces.levels,
                     colors='black', alpha=0.5)
    ax.clabel(cnt, inline=True, fontsize=8, fmt="%.0f")
    ax.scatter(x[:,0], x[:,1], c="black", marker='o', alpha=0.5)
    if highlight_last:
        ax.scatter(x[-1,0], x[-1,1], c='red', marker='o', alpha=0.5)
    ax.set_title("f(x)")
    fig.colorbar(im1, ax=ax, location="bottom", shrink=0.8, pad=0.05)

    ax = fig.add_subplot(132)
    ax.axis('off')
    im2 = ax.imshow(y_mu,
                    extent=[spaces.xmin[0], spaces.xmax[0],
                            spaces.xmin[1], spaces.xmax[1]],
                    vmin=spaces.vmin, vmax=spaces.vmax, alpha=0.8, cmap='RdBu_r')
    cnt = ax.contour(x_plot, y_plot, y_mu, levels=spaces.levels,
                     colors='black', alpha=0.5)
    ax.clabel(cnt, inline=True, fontsize=8, fmt="%.0f")
    if highlight_last:
        ax.scatter(x[-1,0], x[-1,1], c='red', marker='o', alpha=0.5)
    ax.set_title("model")
    fig.colorbar(im2, ax=ax, location="bottom", shrink=0.8, pad=0.05)

    ax = fig.add_subplot(133)
    ax.axis('off')

    scale = "linear"
    ratio = np.max(np.abs(fct))/np.min(np.abs(fct) + 1.0e-5)
    if ratio > 100:
        scale = "symlog"

    im3 = ax.imshow(fct,
                    extent=[spaces.xmin[0], spaces.xmax[0],
                            spaces.xmin[1], spaces.xmax[1]],
                    alpha=0.8, cmap='RdBu_r', norm=scale)
    if highlight_last:
        ax.scatter(x[-1,0], x[-1,1], c='red', marker='o', alpha=0.5)
    ax.set_title(fct_name)
    fig.colorbar(im3, ax=ax, location="bottom", shrink=0.8, pad=0.05)

    plt.savefig(filename, dpi=100)
    plt.close()

def violins_array(filename: str,
                  x: list[ndarray],
                  x_labels: list[str],
                  y_label: str | None=None,
                  title: str | None=None) -> None:
    """
    Generates a violin plot for an array of data.

    This function creates a violin plot to visualize the distribution of
    multiple datasets.

    Args:
        filename: The filename for saving the plot.
        x: A list of NumPy arrays, where each array represents a dataset.
        x_labels: A list of labels for the datasets.
        y_label: The label for the y-axis.
        title: The title of the plot.
    """

    plt.clf()
    fig = plt.figure()
    ax  = fig.add_subplot(111)

    violin = ax.violinplot(x, showmeans=True)
    ax.set_xticks(np.arange(len(x_labels))+1)
    ax.set_xticklabels(x_labels)
    if (y_label is not None): ax.set_ylabel(y_label)

    ratio = np.max(np.abs(x))/np.min(np.abs(x) + 1.0e-5)
    if ratio > 100:
        if np.any(x < 0.0):
            ax.set_yscale("symlog")
        else:
            ax.set_yscale("log")

    if (title is not None): ax.set_title(title)

    plt.savefig(filename, dpi=100)
    plt.close()

def multi_bar(filename: str,
              x: dict[str, list[float]],
              x_labels: list[str],
              bar_labels: list[str],
              y_label: str | None=None,
              title: str | None=None) -> None:
    """
    Generates a multi-bar plot.

    This function creates a bar plot to compare multiple datasets across
    different categories.

    Args:
        filename: The filename for saving the plot.
        x: A dictionary where keys are bar labels and values are lists of data.
        x_labels: A list of labels for the x-axis categories.
        bar_labels: A list of labels for the bars.
        y_label: The label for the y-axis.
        title: The title of the plot.
    """

    dx = np.arange(len(x_labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()

    k = 0
    for m in bar_labels:
        ax.bar(dx + k*width, x[m], width, label=bar_labels[k])
        k += 1

    if (y_label is not None): ax.set_ylabel(y_label)
    if (title is not None): ax.set_title(title)
    ax.set_yscale('log')
    ax.set_xticks(dx)
    ax.set_xticklabels(x_labels)
    ax.legend()

    plt.savefig(filename, dpi=100)
    plt.close()

def map_ints_to_colors(int_list, cmap_name='turbo'):
  """
  Maps a list of integers to a list of RGBA color values using a Matplotlib colormap.

  Args:
    int_list (list or np.array): A list or NumPy array of integers.
    cmap_name (str, optional): The name of the Matplotlib colormap to use.
                                Defaults to 'viridis'.

  Returns:
    list: A list of RGBA tuples, where each tuple corresponds to an integer
          in the input list, mapped through the colormap. Returns an empty
          list if int_list is empty. Returns a list of the 'middle' color
          if all integers in the list are the same.
  """
  if not int_list:
    return []

  # Convert to numpy array for easier min/max handling if needed
  values = np.array(int_list)

  # Create a normalizer instance mapping the integer range to [0, 1]
  # Handles the case where min == max automatically
  norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())

  # Get the colormap instance
  cmap = cm.get_cmap(cmap_name)

  # Apply the normalization and colormap to each integer
  # cmap(norm(value)) returns an RGBA tuple (Red, Green, Blue, Alpha)
  colors = [cmap(norm(value)) for value in values]

  return colors

def scatter_names(filename: str,
                  x: dict[str, float],
                  y: dict[str, float],
                  names: list[str],
                  use_log_scale: bool=True,
                  colors: list[str] | None=None,
                  x_label: str | None=None,
                  y_label: str | None=None,
                  title: str | None=None) -> None:
    """
    Generates a scatter plot with names.

    This function creates a scatter plot where each point is labeled with a
    name.

    Args:
        filename: The filename for saving the plot.
        x: A dictionary where keys are names and values are x-coordinates.
        y: A dictionary where keys are names and values are y-coordinates.
        names: A list of names to plot.
        x_label: The label for the x-axis.
        y_label: The label for the y-axis.
        title: The title of the plot.
    """

    plt.clf()
    fig = plt.figure()
    ax  = fig.add_subplot(111)

    if colors is not None:
        cs = map_ints_to_colors(colors)
        ct = cs
    else:
        cs = ["red"]*len(names)
        ct = ["black"]*len(names)

    texts = []
    for k, m in enumerate(names):
        ax.scatter(x[m], y[m], marker='o', color=cs[k])
        texts.append(ax.text(x[m], y[m], m, fontsize=10, c=ct[k]))

    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))

    if x_label is not None: ax.set_xlabel(x_label)
    if y_label is not None: ax.set_ylabel(y_label)
    if title is not None: ax.set_title(title)
    if use_log_scale: ax.set_yscale("log")

    ax.grid(True)
    fig.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
