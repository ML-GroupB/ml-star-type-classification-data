import matplotlib.animation as animation
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from lib.data_printer import data_plotter, box_plotter


def visualise(data, X, y):
    ### VISUALISATION - Intro ###

    # scatter plot
    scatter_matrix(data, figsize=(20, 14))
    plt.show()

    ### VISUALISATION - Numerical ###

    numerical_features = ['Temperature', 'Luminosity', 'Radius', 'Magnitude']

    # distribution of data for numerical features
    print(data[numerical_features].describe(percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]))
    print()  # newline

    # plot distributions (histogram + boxplot)
    for feature in numerical_features:
        if ['Luminosity', 'Radius'].__contains__(feature):
            box_plotter(feature, data, 'log')
        else:
            box_plotter(feature, data)
        plt.show()

    # calculate correlation using heatmap
    corr_pearson = data[numerical_features].corr(method='pearson')
    corr_spearman = data[numerical_features].corr(method='spearman')
    corr_kendall = data[numerical_features].corr(method='kendall')

    plt.figure(figsize=(20, 5))
    ax1 = plt.subplot(1, 3, 1)
    sns.heatmap(corr_pearson, annot=True, cmap='RdYlGn', vmin=-1, vmax=+1)
    plt.title('Pearson Correlation')

    ax2 = plt.subplot(1, 3, 2, sharex=ax1)
    sns.heatmap(corr_spearman, annot=True, cmap='RdYlGn', vmin=-1, vmax=+1)
    plt.title('Spearman Correlation')

    ax3 = plt.subplot(1, 3, 3, sharex=ax2)
    sns.heatmap(corr_kendall, annot=True, cmap='RdYlGn', vmin=-1, vmax=+1)
    plt.title('Kendall Correlation')
    plt.show()

    # luminosity / radius good good
    # temperature maybe good???

    # check correlations' results

    stars_names = ['Red Dwarf', 'Brown Dwarf', 'White Dwarf', 'Main Sequence', 'Super Giants', 'Hyper Giants']

    data_plotter(X, y, 1, 2, "Stars", stars_names, X.columns[1] + " [Lo = 3.828 x 10^26 Watts]",
                 X.columns[2] + " [Ro = 6.9551 x 10^8 m]", "Lo", "Ro", "log", "log")
    plt.show()
    data_plotter(X, y, 0, 1, "Stars", stars_names, X.columns[0] + " [K]", X.columns[1] + " [Lo = 3.828 x 10^26 Watts]",
                 "K",
                 "Lo", None, "log")
    plt.show()

    ### VISUALISATION - Categorical ###

    categorical_features = ['Color', 'Spectral_Class']

    # plot distribution of categorical features
    for feature in categorical_features:
        plt.figure(figsize=(10, 7))
        data[feature].value_counts().plot(kind='bar')
        plt.title(feature)
        plt.grid()
        plt.show()

    # cross table of features using heatmap
    sns.heatmap(pd.crosstab(data.Color, data.Spectral_Class),
                cmap='RdYlGn',
                annot=True, fmt='.0f')
    plt.figure(figsize=(10, 5))
    plt.show()

    ### VISUALISATION - PCA ###

    pca = data[numerical_features]
    pca = StandardScaler().fit_transform(pca)

    # define 3D PCA and apply PCA
    pc_model = PCA(n_components=3)
    pc = pc_model.fit_transform(pca)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(20, 60)
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=y, marker='o')
    fig.show()

    # optional gif but good visualisation
    gif = False
    if gif:
        def rotate(angle):
            ax.view_init(azim=angle)

        print("Making animation")
        rot_animation = animation.FuncAnimation(fig, rotate, frames=range(0, 360, 1))
        rot_animation.save('rotation.gif', fps=30, dpi=120, writer='imagemagick')

    # define 2D PCA and apply PCA
    pc_model = PCA(n_components=2)
    pc = pc_model.fit_transform(pca)

    ax = fig.add_subplot(111)
    ax.scatter(pc[:, 0], pc[:, 1], c=y, marker='o')
    fig.show()
