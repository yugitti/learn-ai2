import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', 'v')
    colors = ('red', 'blue', 'green', 'black', 'purple')
    marker_colors = ('tomato', 'lightcyan', 'lightgreen', 'gray', 'plum')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:,1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))

    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    z = z.reshape(xx1.shape)
    fig = plt.figure(figsize=(10,10),dpi=100)
    ax1 = fig.add_subplot(2,1,1)
    ax1.contourf(xx1, xx2, z, alpha=0.3, cmap = cmap)

    ax1.set_xlim(xx1.min(), xx1.max())
    ax1.set_ylim(xx2.min(), xx2.max())

    for idx, c1 in enumerate(np.unique(y)):
        ax1.scatter(x=X[y == c1, 0], y=X[y == c1, 1],
                   alpha = 0.8,
                   c = marker_colors[idx],
                   marker = markers[idx],
                   label = c1,
                   edgecolor = 'white')

    if test_idx:

        X_test, y_test = X[test_idx, :], y[test_idx]
        ax1.scatter(X_test[:, 0], X_test[:,1],
                   c = '',
                   edgecolor = 'white',
                   alpha = 1.0,
                   linewidth = 1,
                   marker = 'o',
                   s = 100,
                   label = 'test set')
