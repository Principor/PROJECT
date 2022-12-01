import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def k_moving_avg(x, y, k):
    return x[k-1:], np.convolve(y, np.ones(k), 'valid') / k


def plot():
    files = {
        "lr=0.001": "ppo_lr0.001",
        "lr=0.0003": "ppo_lr0.0003",
        "lr=0.0001": "ppo_lr0.0001",
        "lr=0.00003": "ppo_lr0.00003"
    }

    for name, path in files.items():
        data = pd.read_json("../summaries/json/{}.json".format(path))
        x = np.array(data[1])
        y = np.array(data[2])
        x, y = k_moving_avg(x, y, 50)

        plt.plot(x, y, label=name)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot()
