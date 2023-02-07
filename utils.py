import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np

# [None, None] == .unsqueeze(0).unsqueeze(0)
def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)

def read_image_test(path0, path1, segmentation):
    torch.cuda.empty_cache()
    image0 = cv2.imread(path0, cv2.IMREAD_GRAYSCALE)
    image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)

    if image0 is None or image1 is None:
        return None, None, None, None, None, None

    asset = image0
    img = image1

    h1, w1 = image1.shape[:2]
    h0, w0 = image0.shape[:2]
    l_image0 = []
    l_image0_pos = []
    num_vertical_windows = 1
    num_horizontal_windows = 1

    segment_factor = 3
    segment_factor2 = 2

    if segmentation and (h1 <= int(math.floor(h0/segment_factor)) or w1 <= int(math.floor(w0/segment_factor))):
        num_vertical_windows = max(1, int(math.ceil(h0/h1/segment_factor2)))
        # print('num_vertical_windows', num_vertical_windows)
        num_horizontal_windows = max(1, int(math.ceil(w0/w1/segment_factor2)))
        # print('num_horizontal_windows', num_horizontal_windows)
        for i in range(num_vertical_windows):
            h_start = i * h1 * segment_factor2
            h_end = min(h_start + segment_factor * h1, h0)
            for j in range(num_horizontal_windows):
                w_start = j*w1*segment_factor2
                w_end = min(w_start+segment_factor*w1, w0)
                if h_start != h0 and w_start != w0:
                    l_image0.append(image0[h_start:h_end, w_start:w_end])
                    l_image0_pos.append((h_start, w_start))
    else:
        l_image0.append(image0)
        l_image0_pos.append((0, 0))


    if min(h1, w1) <= 64:
       scale0 = 128 / min(h1, w1)
    else:
       scale0 = 1

    l_resized_image0 = []
    # scale_factor = max(num_horizontal_windows, num_vertical_windows)
    if scale0 != 1:
        for crop in l_image0:
            h_crop, w_crop = crop.shape[:2]
            l_resized_image0.append(cv2.resize(crop,(int(w_crop*scale0), int(h_crop*scale0))))
            # l_resized_image0.append(crop)
        image1 = cv2.resize(image1, (int(round(w1 * scale0)), int(round(h1 * scale0))))
    else:
        for crop in l_image0:
            l_resized_image0.append(crop)

    resized_h, resized_w = l_resized_image0[0].shape[:2]
    if image1.shape[0] < resized_h:
        vertical_padding = int((resized_h - image1.shape[0]) / 2)
        image1 = cv2.copyMakeBorder(image1, vertical_padding, vertical_padding, 0, 0, cv2.BORDER_CONSTANT)

    if image1.shape[1] < resized_w:
        horizontal_padding = int((resized_w - image1.shape[1]) / 2)
        image1 = cv2.copyMakeBorder(image1, 0, 0, horizontal_padding, horizontal_padding, cv2.BORDER_CONSTANT)

    l_image1 = []
    image1_flip = cv2.flip(image1, 1)
    l_image1.append(image1)
    l_image1.append(cv2.rotate(image1, cv2.ROTATE_90_CLOCKWISE))
    l_image1.append(cv2.rotate(image1, cv2.ROTATE_180))
    l_image1.append(cv2.rotate(image1, cv2.ROTATE_90_COUNTERCLOCKWISE))
    l_image1.append(image1_flip)
    l_image1.append(cv2.rotate(image1_flip, cv2.ROTATE_90_CLOCKWISE))
    l_image1.append(cv2.rotate(image1_flip, cv2.ROTATE_180))
    l_image1.append(cv2.rotate(image1_flip, cv2.ROTATE_90_COUNTERCLOCKWISE))

    # l_inp1 = []
    # for padded_img in l_image1:
    #     l_inp1.append(frame2tensor(padded_img, device))
    #
    # l_inp0 = []
    # for resized_crop in l_resized_image0:
    #     l_inp0.append(frame2tensor(resized_crop, device))

    return l_resized_image0, l_image1, l_image0_pos, scale0, asset, img

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


def plot_keypoints(kpts0, kpts1, color='w', ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_matches(asset_idx, l_image0_pos, scale0, show_selection, kpts0, kpts1, color, lw=1.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()

    # Delete outliers by Z-score
    mean_p = np.mean(kpts0, axis=0)

    man_dist = []
    for point in kpts0:
        man_dist.append(math.sqrt(math.pow(point[0]-mean_p[0],2) + math.pow(point[1]-mean_p[1],2)))
    mean_dist = np.mean(man_dist)
    std_dist = np.std(man_dist)

    outlier = []
    threshold = 3
    for i in range(len(man_dist)):
        z = (man_dist[i] - mean_dist) / std_dist
        if z > threshold:
            outlier.append(i)

    # Discard outliers
    l_kpts0 = kpts0.tolist()
    for idx in outlier:
        del l_kpts0[idx]
    kpts0 = np.array(l_kpts0)

    # Revert coordinates
    l_ori_kpts0 = []
    for point in kpts0:
        l_ori_kpts0.append([l_image0_pos[asset_idx][1] + point[0]/scale0, l_image0_pos[asset_idx][0] + point[1]/scale0])
    ori_kpts0 = np.array(l_ori_kpts0)

    # Select image in asset
    min_p = np.amin(ori_kpts0, axis=0)
    max_p = np.amax(ori_kpts0, axis=0)
    print("Suggested top left coordiantes = ", (min_p[0], min_p[1]))
    print("Suggested bottom right coordiantes = ", (max_p[0], max_p[1]))

    # Show selection
    if show_selection:
        fkpts0 = transFigure.transform(ax[0].transData.transform(ori_kpts0))
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


    # Show center
    # mean_p = np.mean(kpts0, axis=0)
    # ori_mean_p_x = l_image0_pos[asset_idx][1] + mean_p[0]/scale0
    # ori_mean_p_y = l_image0_pos[asset_idx][0] + mean_p[1]/scale0
    # ax[0].scatter(ori_mean_p_x, ori_mean_p_y, c='r', s=ps)


def make_matching_plot(max_pair, color, show_keypoints, show_selection, l_image0_pos, scale0):
    kpts0, kpts1, mkpts0, mkpts1, asset_idx, asset, img = max_pair
    plot_image_pair([asset, img])
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color='w', ps=4)
        plot_keypoints(kpts0, kpts1, color='w', ps=2)
    plot_matches(asset_idx, l_image0_pos, scale0, show_selection, mkpts0, mkpts1, color)

    plt.show()