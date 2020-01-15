import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt


'''
t is number of standard deviations
'''

def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    """Return an axes of confidence bands using a simple approach.

    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

    References
    ----------
    .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb

    """
    if ax is None:
        ax = plt.gca()

    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", edgecolor="")

    return ax

'''
x2, y2 are the modelling response. 

x2: linspace from min(x) to max(x)
y2: conditional mean response
X: the actual data (needed to compute the standard deviation) 
t: width of the CI, in std. devs.
'''
def plot_ci_normal_dist(t, x2, y2, means, ax=None, color="#b9cfe7"):
    import matplotlib
    from matplotlib import colors

    new_colour = colors.to_rgba(color, alpha=0.23)
    print()
    """Return an axes of confidence bands using a simple approach.

    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

    References
    ----------
    .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb

    """
    # we can compute the stddev via a built in, or explicitly.
    # let's try it explicitly
    # assert means.shape[1] == X.shape[1] == 25
    # assert means.shape[0] == 1
    from matplotlib import cm
    means = means.reshape((-1, len(means)))
    std_devs = np.sqrt(means * (100-means)/25)

    ci = t*std_devs

    if ax is None:
        ax = plt.gca()

    ci = ci.squeeze()
    # print(matplotlib.colors.cnames[color])
    ax.fill_between(x2, y2 + ci, y2 - ci, color=new_colour, edgecolor="")

    return ax

def plot_ci_bootstrap(xs, ys, resid, nboot=500, ax=None):
    """Return an axes of confidence bands using a bootstrap approach.

    Notes
    -----
    The bootstrap approach iteratively resampling residuals.
    It plots `nboot` number of straight lines and outlines the shape of a band.
    The density of overlapping lines indicates improved confidence.

    Returns
    -------
    ax : axes
        - Cluster of lines
        - Upper and Lower bounds (high and low) (optional)  Note: sensitive to outliers

    References
    ----------
    .. [1] J. Stults. "Visualizing Confidence Intervals", Various Consequences.
       http://www.variousconsequences.com/2010/02/visualizing-confidence-intervals.html

    """
    if ax is None:
        ax = plt.gca()

    bootindex = sp.random.randint

    for _ in range(nboot):
        resamp_resid = resid[bootindex(0, len(resid) - 1, len(resid))]
        # Make coeffs of for polys
        pc = sp.polyfit(xs, ys + resamp_resid, 1)
        # Plot bootstrap cluster
        ax.plot(xs, sp.polyval(pc, xs), "b-", linewidth=2, alpha=3.0 / float(nboot))

    return ax

'''
x is simply a lin space, (1 to T). 
Y is the entire batch of accuracies, i.e. N x T (N is the number of samples, T is the number of timesteps)

We use the simplest method for plotting the classifier performance: just compute the standard deviation at each timestep.

To make it a "plot", we also fit a simple regression curve. 

The exact methodology:
- compute the mean acc at each timestep
- fit a 1D polynomial (regression) for the mean
- compute the standard deviations around the mean, at each timestep
- fill in the area between the +2/-2 deviations around the mean 

'''


'''
loads data, for graphing purposes
'''
def load_data(path, pattern="kl_penalty"):
    # we can glob the entire path
    #
    import os
    import numpy as np

    all_accs = np.zeros((25,100))
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if pattern in dir:
                print(dir)

                ind = int(dir.split("_")[-1])
                print(ind)

                with open(os.path.join(root, dir,"accs.txt"), "r") as file:
                    counter = 0
                    for line in (file):
                        if ";" in line:
                            if counter==100:
                                print(ind, counter)
                            # print(counter)
                            acc = line.split(";")[0]
                            all_accs[ind,counter] = float(acc)
                            counter+=1


                    # break
                    # print(file.readlines(1))
    # print(all_accs)
    print(all_accs.shape)
    #                 open with w => overwrite!
    return all_accs
    pass

def load_data_baselines(path,  pattern="kl_penalty", mode="kl_penalty"):
    # we can glob the entire path
    #
    import os
    import numpy as np

    all_accs = np.zeros((25,100))
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            print(root, dirs)

            if pattern in dir and root==path:
                # print(dir)

                ind = int(dir.split("_")[-1])
                # print(ind)

                if mode == "kl_penalty":
                    with open(os.path.join(root, dir,"accs.txt"), "r") as file:
                        counter = 0
                        for line in (file):
                            if ";" in line:
                                if counter==100:
                                    print(ind, counter)
                                # print(counter)
                                acc = line.split(";")[0]
                                all_accs[ind,counter] = float(acc)
                                counter+=1
                elif mode == "uncertainty" or mode == "random":
                    # print(dir)
                    #
                    # print(os.path.join(root, dir, mode + "_current_accs.txt"))
                    with open(os.path.join(root, dir, mode + "_current_accs.txt"), "r") as file:
                        counter = 0
                        for line in (file):
                            if " " in line:
                                if counter==100:
                                    print(ind, counter)
                                # print(counter)
                                acc = line.split(" ")[0]
                                all_accs[ind,counter] = float(acc)
                                counter+=1


                    # break
                    # print(file.readlines(1))
    # print(all_accs)
    print(all_accs.shape)
    #                 open with w => overwrite!
    return all_accs
    pass



def stddev_plot(x,y):
    fig,ax = plt.subplots()

    ax.plot(x,y)
    fig.show()

    pass


def gen_ci_plot(accs, fig, ax, color="g"):
    x = np.arange(0, accs.shape[1])
    y = np.mean(accs, axis=0)
    t = 2

    # Modeling with Numpy
    def equation(a, b):
        """Return a 1D polynomial."""
        return np.polyval(a, b)

    p, cov = np.polyfit(x, y, 1, cov=True)  # parameters and covariance from of the fit of 1-D polynom.
    y_model = equation(p, x)  # model using the fit parameters; NOTE: parameters here are coefficients
    # Plotting --------------------------------------------------------------------
    # fig, ax = plt.subplots(figsize=(8, 6))
    # Data
    ax.plot(
        x, y, "o", color=color, markersize=8,
        markeredgewidth=1, markeredgecolor=color, markerfacecolor="None",
    )
    # Fit
    ax.plot(x, y_model, "-", color=color, linewidth=1.5, alpha=0.5, label="r={}".format(p))
    x2 = np.linspace(np.min(x), np.max(x), len(x))
    y2 = equation(p, x2)
    # Confidence Interval (select one)
    # plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax)
    # plot_ci_bootstrap(x, y, resid, ax=ax)

    means = y
    # means = means.reshape((-1, len(means)))
    std_devs = np.sqrt(means * (100 - means) / 25)
    std_vars = means * (100 - means) / 25

    # ax.plot(x, std_vars, label="std_vars", color=color)

    plot_ci_normal_dist(t, x2, y2, y, ax=ax, color=color)
    # # Prediction Interval
    # pi = t * s_err * np.sqrt(1 + 1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    # ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
    # ax.plot(x2, y2 - pi, "--", color="0.5", label="95% Prediction Limits")
    # ax.plot(x2, y2 + pi, "--", color="0.5")
    # Figure Modifications --------------------------------------------------------
    # Borders
    ax.spines["top"].set_color("0.5")
    ax.spines["bottom"].set_color("0.5")
    ax.spines["left"].set_color("0.5")
    ax.spines["right"].set_color("0.5")
    ax.get_xaxis().set_tick_params(direction="out")
    ax.get_yaxis().set_tick_params(direction="out")
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    # Labels
    plt.title("Fit Plot for Weight", fontsize="14", fontweight="bold")
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.xlim(np.min(x) - 1, np.max(x) + 1)
    # Custom legend
    handles, labels = ax.get_legend_handles_labels()
    display = (0, 1)
    anyArtist = plt.Line2D((0, 1), (0, 0), color=color)  # create custom artists
    legend = plt.legend(
        [handle for i, handle in enumerate(handles) if i in display] + [anyArtist],
        [label for i, label in enumerate(labels) if i in display] + ["95% Confidence Limits"],
        loc=9, bbox_to_anchor=(0, -0.21, 1., 0.102), ncol=3, mode="expand"
    )
    frame = legend.get_frame().set_edgecolor("0.5")
    # Save Figure
    plt.tight_layout()
    fig.legend()

    plt.savefig("filename.png", bbox_extra_artists=(legend,), bbox_inches="tight")
    fig.show()
    return fig, ax


if __name__ == "__main__":
    accs = load_data("/scratch/gobi1/johnchen/vaal_results")
    random_accs = load_data_baselines("/scratch/gobi1/johnchen/vaal_results", mode="random")
    uncertainty_accs = load_data_baselines("/scratch/gobi1/johnchen/vaal_results", mode="uncertainty")

    # accs = accs[:,:50]
    # random_accs = random_accs[:,:50]
    # uncertainty_accs = uncertainty_accs[:,:50]

    # Computations ----------------------------------------------------------------
    # Raw Data

    '''trying the normal equation line fit'''

    '''
    x = np.arange(0,all_accs.shape[1])
    x = np.reshape(x,(1,100))
    x = np.repeat(x, 25, axis=0)
    
    y = all_accs

    '''
    '''
    Couple approaches: either normal equation line fit. Or, we can do just on the mean
    '''

    '''trying the regular mean fit'''
    fig, ax = plt.subplots(figsize=(8, 6))

    #
    # ax.set_color_cycle(['red', 'black', 'yellow'])
    # fig, ax = gen_ci_plot(accs, fig, ax, color="g")
    # fig, ax = gen_ci_plot(random_accs, fig, ax, color="r")
    # fig, ax = gen_ci_plot(uncertainty_accs, fig, ax, color="b")

    fig, ax = gen_ci_plot(accs, fig, ax, color="g")
    fig, ax = gen_ci_plot(random_accs, fig, ax, color="r")
    fig, ax = gen_ci_plot(uncertainty_accs, fig, ax, color="b")

    pass
