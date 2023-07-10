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
def plot_avg(data, filename, avg_type):

    stp   = data[:,0]
    avg   = data[:,1]
    p     = data[:,2]
    m     = data[:,3]

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

    fig, ax = plt.subplots(1,1,figsize=(5,5))
    fig.suptitle(filename)

    ax.set_xlabel('evaluations')
    ax.set_ylabel(ylabel)
    plt.yscale(avg_type)
    plt.plot(stp, avg, color='blue', label='avg')
    plt.fill_between(stp, p, m,
                    alpha=0.4,
                    color='blue',
                    label="+/- std")
    plt.grid(True)
    plt.legend()

    fig.tight_layout()
    fig.savefig(filename+'.png')
