import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np


# Parameters
n_classes = 5
plot_colors = "rybwg"
plot_step = 0.02  # Load data
data = pd.read_csv("classified5_clean.csv")

for pairidx, pair in enumerate([[0, 3], [0, 4], [0, 5],
                                [1, 2], [1, 3], [2, 3]]):
    # We only take the two corresponding features
    X = data[:, pair]
    y = data["consumption"]

    # Train
    clf = DecisionTreeClassifier().fit(X, y)

    # Plot the decision boundary
    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    # plt.xlabel(iri.feature_names[pair[0]])
    # plt.ylabel(iris.feature_names[pair[1]])

    # Plot the training points for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(y == i)
    # label=iris.target_names[i]
    plt.scatter(X[idx, 0], X[idx, 1], c=color,
                cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
