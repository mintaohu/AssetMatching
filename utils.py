import numpy as np
import math
from matplotlib import pyplot as plt
import matplotlib


# Detect outliers by z-score iteration
def discard_outliers(kpts0):
    kpts0 = np.array(kpts0)

    if len(kpts0) == 1 or len(kpts0) == 0:
        return kpts0, True

    # Delete outliers by Z-score
    mean_p = np.mean(kpts0, axis=0)

    man_dist = []
    for point in kpts0:
        man_dist.append(math.sqrt(math.pow(point[0] - mean_p[0], 2) + math.pow(point[1] - mean_p[1], 2)))

    mean_dist = np.mean(man_dist)
    std_dist = np.std(man_dist)

    if std_dist == 0:
        return kpts0, True

    outlier = []
    threshold = 0
    for i in range(len(man_dist)):
        z = (man_dist[i] - mean_dist) / std_dist
        if z > threshold:
            outlier.append(i)

    # Discard outliers
    l_kpts0 = kpts0.tolist()
    for idx in reversed(outlier):
        del l_kpts0[idx]
    kpts0 = np.array(l_kpts0)

    if len(kpts0) == 1:
        return kpts0, True

    return kpts0, False


def plot_image_pair(imgs, dpi=100, size=6, pad=.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size*n, size*3/4) if size is not None else None
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)


def plot_matches(show_selection, kpts0, single_point, lw=1.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()
    transFigure = fig.transFigure.inverted()

    # Scatter all matching points
    # ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c='r', s=ps)

    # Scatter single matching point
    if single_point:
        ax[0].scatter(kpts0[0][0], kpts0[0][1], c='r', s=ps)
        plt.show()
        return

    # Show selection
    if show_selection:
        min_p = np.amin(kpts0, axis=0)
        max_p = np.amax(kpts0, axis=0)
        print("Suggested top left coordinates = ", (min_p[0], min_p[1]))
        print("Suggested bottom right coordnantes = ", (max_p[0], max_p[1]))

        fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
        fmin_p  = np.amin(fkpts0, axis=0)
        fmax_p = np.amax(fkpts0, axis=0)

        lines = []
        lines.append(matplotlib.lines.Line2D((fmin_p[0], fmax_p[0]), (fmin_p[1], fmin_p[1]), zorder=1, transform=fig.transFigure, c='r', linewidth=lw))
        lines.append(
            matplotlib.lines.Line2D((fmin_p[0], fmax_p[0]), (fmax_p[1], fmax_p[1]), zorder=1, transform=fig.transFigure,
                                    c='r', linewidth=lw))
        lines.append(
            matplotlib.lines.Line2D((fmin_p[0], fmin_p[0]), (fmin_p[1], fmax_p[1]), zorder=1, transform=fig.transFigure,
                                    c='r', linewidth=lw))
        lines.append(
            matplotlib.lines.Line2D((fmax_p[0], fmax_p[0]), (fmin_p[1], fmax_p[1]), zorder=1, transform=fig.transFigure,
                                    c='r', linewidth=lw))
        fig.lines = lines

    # Scatter center
    else:
        mean_p = np.mean(kpts0, axis=0)
        ax[0].scatter(mean_p[0],mean_p[1], c='r', s=ps)

    plt.show()
    return

# Read image info from plist
def find_image(coordinate_str):
    digits = []
    digit = ''
    for char in coordinate_str:
        if char.isdigit():
            digit += char
        elif len(digit) != 0:
            digits.append(int(digit))
            digit = ''
        else:
            digit = ''

    return digits
