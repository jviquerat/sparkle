# Generic imports
import numpy             as np
import matplotlib.pyplot as plt

# Custom imports
from sparkle.src.utils.error import error

plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['figure.titleweight'] = 'bold'

# Plot averaged fields
def plot_avg(data, filename, avg_type):

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

# Return avg, p and m fields depending on avg type
def return_plottables(avg, p, m, avg_type):

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

# 1D rendering for regular trainer
def render_1D_regular(filename, spaces, x, c, x_plot, cost_map):

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

# 2D rendering for regular trainer
def render_2D_regular(filename, spaces, x, c, x_plot, y_plot, cost_map):

    plt.clf()
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    fig.set_size_inches(3, 3)
    fig.subplots_adjust(0,0,1,1)

    ax.set_xlim([spaces.xmin[0], spaces.xmax[0]])
    ax.set_ylim([spaces.xmin[1], spaces.xmax[1]])
    ax.axis('off')
    ax.imshow(cost_map,
              extent=[spaces.xmin[0], spaces.xmax[0],
                      spaces.xmin[1], spaces.xmax[1]],
              vmin=spaces.vmin, vmax=spaces.vmax,
              alpha=0.8, cmap='RdBu_r')

    cnt = ax.contour(x_plot, y_plot, cost_map, levels=spaces.levels,
                     colors='black', alpha=0.5)
    ax.clabel(cnt, inline=True, fontsize=8, fmt="%.0f")
    ax.scatter(x[:,0], x[:,1], c="black", marker='o', alpha=0.8)

    plt.savefig(filename, dpi=100)
    plt.close()

# 1D rendering for metamodel trainer
def render_1D_metamodel(filename, spaces, x, c,
                        x_plot, cost_map,
                        y_mu, y_std, fct, fct_name,
                        highlight_last=True):

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
    ax.plot(x_plot, fct, color='r')
    ax.grid()
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_ylabel(fct_name)

    fig.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()

# 2D rendering for metamodel trainer
def render_2D_metamodel(filename, spaces, x, c,
                        x_plot, y_plot, cost_map,
                        y_mu, y_std, fct, fct_name,
                        highlight_last=True):

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
    ax.imshow(cost_map,
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

    ax = fig.add_subplot(132)
    ax.axis('off')
    ax.imshow(y_mu,
              extent=[spaces.xmin[0], spaces.xmax[0],
                      spaces.xmin[1], spaces.xmax[1]],
              vmin=spaces.vmin, vmax=spaces.vmax, alpha=0.8, cmap='RdBu_r')
    cnt = ax.contour(x_plot, y_plot, y_mu, levels=spaces.levels,
                     colors='black', alpha=0.5)
    ax.clabel(cnt, inline=True, fontsize=8, fmt="%.0f")
    if highlight_last:
        ax.scatter(x[-1,0], x[-1,1], c='red', marker='o', alpha=0.5)
    ax.set_title("model")

    ax = fig.add_subplot(133)
    ax.axis('off')
    ax.imshow(fct,
              extent=[spaces.xmin[0], spaces.xmax[0],
                      spaces.xmin[1], spaces.xmax[1]],
              alpha=0.8, cmap='RdBu_r')
    if highlight_last:
        ax.scatter(x[-1,0], x[-1,1], c='red', marker='o', alpha=0.5)
    ax.set_title(fct_name)

    plt.savefig(filename, dpi=100)
    plt.close()

# Violin plot array
def violins_array(filename, x, x_labels, y_label=None, title=None):

    plt.clf()
    fig = plt.figure()
    ax  = fig.add_subplot(111)

    violin = ax.violinplot(x, showmeans=True)
    ax.set_xticks(np.arange(len(x_labels))+1)
    ax.set_xticklabels(x_labels)
    if (y_label is not None): ax.set_ylabel(y_label)
    ax.set_yscale('log')
    if (title is not None): ax.set_title(title)

    plt.savefig(filename, dpi=100)
    plt.close()

# Multi-bar plot
def multi_bar(filename, x, x_labels, bar_labels, y_label=None, title=None):

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

# Scatter plot with names
def scatter_names(filename, x, y, names, x_label=None, y_label=None, title=None):

    plt.clf()
    fig = plt.figure()
    ax  = fig.add_subplot(111)

    for m in names:
        ax.scatter(x[m], y[m], marker='o', color='red')
        ax.text(x[m], y[m], m, fontsize=9, color="red")

    if (x_label is not None): ax.set_xlabel(x_label)
    if (y_label is not None): ax.set_ylabel(y_label)
    if (title is not None): ax.set_title(title)
    ax.set_yscale('log')
    ax.grid(True)
    plt.savefig(filename, dpi=100)
    plt.close()
