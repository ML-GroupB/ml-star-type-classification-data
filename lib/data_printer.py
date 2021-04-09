import matplotlib.pyplot as plt


def data_plotter(X, y, x_axis, y_axis, title, target_names, x_label=None, y_label=None):
    if x_label is not None:
        plt.xlabel(x_label)
    if x_label is not None:
        plt.ylabel(y_label)

    formatter = plt.FuncFormatter(lambda i, *args: target_names[int(i)])
    plt.title(label=title)
    plt.scatter(X[:, x_axis], X[:, y_axis], c=y)
    plt.colorbar(format=formatter)
    plt.show()



