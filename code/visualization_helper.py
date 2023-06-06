import matplotlib.pyplot as plt
from pathlib import Path
import os

def setup():
    # import seaborn as sns
    # sns.set()
    params = {
        "ytick.color": "black",
        "xtick.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Serif"],
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.framealpha": 0.0,
        "legend.frameon": False,
    }
    plt.rcParams.update(params)


def get_folder():
    file_path = os.path.realpath(__file__)
    assets_path = Path(file_path).parent.parent / 'assets'
    assets_path.mkdir(parents=True, exist_ok=True)
    return assets_path

