import matplotlib.animation as animation
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import EngFormatter
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def visualise(data):
    groups = data.groupby("Type")

    stars_names = {0: 'Red Dwarf', 1: 'Brown Dwarf', 2: 'White Dwarf', 3: 'Main Sequence', 4: 'Super Giants',
                   5: 'Hyper Giants'}
    stars_colors = {0: 'lightcoral', 1: 'sandybrown', 2: 'paleturquoise', 3: 'royalblue', 4: 'mediumpurple',
                    5: 'orchid'}

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
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(15, 12))
        sns.histplot(
            ax=ax0,
            x=data[feature],
            bins=100
        )
        ax0.grid()
        ax0.set_title(feature)

        sns.boxplot(ax=ax1, x=data[feature], width=0.25)
        ax1.grid()
        ax1.set_title(feature + ' - boxplot')

        if ['Luminosity', 'Radius'].__contains__(feature):
            ax0.set_xscale("log")
            ax1.set_xscale("log")

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

    plt.subplot(1, 3, 3, sharex=ax2)
    sns.heatmap(corr_kendall, annot=True, cmap='RdYlGn', vmin=-1, vmax=+1)
    plt.title('Kendall Correlation')
    plt.show()

    # luminosity / radius good good
    # temperature maybe good???

    # check correlations' results

    # luminosity/radius
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for type, group in groups:
        ax.scatter(group["Luminosity"], group["Radius"], c=stars_colors[type], label=stars_names[type])

    plt.xlabel("Luminosity [Lo = 3.828 x 10^26 Watts]")
    plt.ylabel("Radius [Ro = 6.9551 x 10^8 m]")

    ax.xaxis.set_major_formatter(EngFormatter(unit="Lo"))
    ax.yaxis.set_major_formatter(EngFormatter(unit="Ro"))

    ax.set(xscale="log", yscale="log")

    plt.legend(loc='lower right')
    plt.title(label="Stars")
    fig.show()

    # temperature/luminosity
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for type, group in groups:
        ax.scatter(group["Temperature"], group["Luminosity"], c=stars_colors[type], label=stars_names[type])

    plt.xlabel("Temperature [K]")
    plt.ylabel("Luminosity [Lo = 3.828 x 10^26 Watts]")

    ax.xaxis.set_major_formatter(EngFormatter(unit="K"))
    ax.yaxis.set_major_formatter(EngFormatter(unit="Lo"))

    ax.set(yscale="log")

    plt.legend(loc='lower right')
    plt.title(label="Stars")
    fig.show()

    ### VISUALISATION - Categorical ###

    categorical_features = ['Color', 'Spectral_Class']

    # plot distribution of categorical features
    for feature in categorical_features:
        plt.figure(figsize=(15, 7))
        sns.countplot(x=data[feature])
        plt.title(feature)
        plt.grid()
        plt.show()

    # cross table of features using heatmap
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.heatmap(pd.crosstab(data.Color, data.Spectral_Class),
                cmap='RdYlGn',
                annot=True, fmt='.0f')
    plt.subplots_adjust(left=0.25)
    fig.show()
    plt.subplots_adjust(left=0)

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

    for g in np.unique(data["Type"]):
        i = np.where(data["Type"] == g)
        ax.scatter(pc[i, 0], pc[i, 1], pc[i, 2], c=stars_colors[g], label=stars_names[g], depthshade=False)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.00), framealpha=1)
    plt.title("3D PCA")
    ax.view_init(20, 60)
    fig.show()

    # optional gif but good visualisation
    gif = False
    if gif:
        def rotate(angle):
            ax.view_init(azim=angle)

        print("Making animation")
        rot_animation = animation.FuncAnimation(fig, rotate, frames=range(0, 360, 1))
        rot_animation.save('rotation.gif', fps=30, dpi=240, writer='imagemagick')

    # define 2D PCA and apply PCA
    pc_model = PCA(n_components=2)
    pc = pc_model.fit_transform(pca)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    for g in np.unique(data["Type"]):
        i = np.where(data["Type"] == g)
        ax.scatter(pc[i, 0], pc[i, 1], c=stars_colors[g], label=stars_names[g])
    ax.legend(loc='lower right', bbox_to_anchor=(1.1, -0.03), framealpha=1)
    plt.title("2D PCA")
    fig.show()
