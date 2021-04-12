import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter


def data_plotter(X, y, x_axis, y_axis, title, target_names, x_label=None, y_label=None, x_unit=None, y_unit=None, scale_x=None, scale_y=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # labels
    if x_label is not None:
        plt.xlabel(x_label)
    if x_label is not None:
        plt.ylabel(y_label)

    # units
    if x_unit is not None:
        ax.xaxis.set_major_formatter(EngFormatter(unit=x_unit))
    if y_unit is not None:
        ax.yaxis.set_major_formatter(EngFormatter(unit=y_unit))

    # scale
    if scale_x is not None:
        ax.set_xscale(scale_x)
    if scale_y is not None:
        ax.set_yscale(scale_y)

    formatter = plt.FuncFormatter(lambda i, *args: target_names[int(i)])
    plt.title(label=title)
    plt.scatter(X.iloc[:, x_axis], X.iloc[:, y_axis], c=y)
    plt.colorbar(format=formatter)


def box_plotter(feature, data, scale=None):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 7))
    if scale is not None:
        ax0.set_xscale(scale)
        ax1.set_xscale(scale)
    ax0.set_xscale('log')
    ax0.hist(data[feature], bins=50)
    ax0.grid()
    ax0.set_title(feature)
    ax1.boxplot(data[feature], vert=False)
    ax1.grid()
    ax1.set_title(feature + ' - boxplot')

