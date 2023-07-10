# Generic imports
import numpy             as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['figure.titleweight'] = 'bold'

# Plot averaged fields
def plot(data, filename, avg_type):

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
    plt.fill_between(stp, avg_p, avg_m,
                    alpha=0.4,
                    color='blue')
    plt.fill_between(stp, bst_p, bst_m,
                    alpha=0.4,
                    color='red')
    plt.grid(True)
    plt.legend()

    fig.tight_layout()
    fig.savefig(filename+'.png')

# Return avg, p and m fields depending on avg type
def return_plottable(avg, p, m, avg_type):

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

    return avg, p, m, label
