import random
import pandas as pd
import matplotlib.pyplot as plt

_, ax = plt.subplots(3, 2)
ax[2, 1].set_xticks([])
ax[2, 1].set_yticks([])

CNT = [0]


def plot_decision_boundary(data, label, title=None):
    cnt = CNT[0]
    positive_data = data[label == 1]
    negative_data = data[label == 0]
    p_ax = ax[cnt//2, cnt%2]
    p_ax.scatter(positive_data['x'], positive_data['y'], marker='o', color='red')
    p_ax.scatter(negative_data['x'], negative_data['y'], marker='x', color='blue')
    p_ax.set_title(title)
    p_ax.set_xticks([])
    p_ax.set_yticks([])
    CNT[0] = cnt + 1


def get_train():
    x = []
    y = []
    label = []

    for i in range(500):
        a = random.random() * 10
        b = random.random() * 10

        x.append(a)
        y.append(b)

        if a - b > 2:
            label.append(1)
        else:
            label.append(0)

    data = pd.DataFrame({"x": x, "y": y})
    label = pd.Series(label)
    return data, label


def get_test():
    x = []
    y = []
    for i in range(50000):
        a = random.random() * 10
        b = random.random() * 10

        x.append(a)
        y.append(b)

    data = pd.DataFrame({"x": x, "y": y})
    return data
