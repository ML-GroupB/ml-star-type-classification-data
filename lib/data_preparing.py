import os
import tarfile
import urllib.request

import matplotlib.pyplot as plt
import sklearn


def download_file(url, path):
    """
    Download file from web and save to file.

    :param url: Any url with file: "http://example.org/file.ext".
    :param path: Path under script directory: "dir/file.ext".
    """
    dir = os.path.dirname(path)
    os.makedirs(dir, exist_ok=True)
    urllib.request.urlretrieve(url, path)


def untar(path):
    """
    Untar file in it's current directory.

    :param path: Path to file.
    """
    assert tarfile.is_tarfile(path), "File \"" + path + "\" is not tar archive!"
    tgz = tarfile.open(path)
    tgz.extractall(os.path.dirname(path))
    tgz.close()


def save_fig(path, dpi=300, tight_layout=True):
    """
    Save matplotlib figure to file.

    :param path: Path to file.
    :param dpi: (Optional) DPI where: 6.4 inches * dpi = pixels
    :param tight_layout: Adjust the padding between and around subplots.
    """
    dir = os.path.dirname(path)
    os.makedirs(dir, exist_ok=True)
    filename = os.path.basename(path)
    extension = filename.split(".")[-1]
    path = os.path.join(dir, filename)
    print("Saving figure", filename)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=extension, dpi=dpi)


def test_split(data, test_size=0.3):
    """
    Split data to train and test subsets.

    :param data: Array like list.
    :param test_size: Size of test subset.
    :return: Returns tuple of matrix like subsets (train_subset, test_subset).
    """
    return sklearn.model_selection.train_test_split(data, test_size)

# Example
# mpath = "help/my.tgz"
# download_file("https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz", mpath)
# untar(mpath)
#
# housing = pd.read_csv("help/housing.csv")
# housing.hist(figsize=(30, 20))
# save_fig("img/my_image.png")


