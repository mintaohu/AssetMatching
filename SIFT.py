import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
import time
from PIL import Image

from utils import discard_outliers, plot_image_pair, plot_matches, find_image


# SIFT(find key points) + FLANN(fast keypoints match) + Z-score iteration(discard outliers)
def feature_matching(uniqueness, outlier_threshold, num_iterations, asset, img, show_match):

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # FLANN parameters and initialize
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Read in images from a filepath as graycsale.
    try:
        im0 = Image.open(asset)
    except:
        print('ASSET NOT FOUND')
        return 'ASSET NOT FOUND'

    try:
        im1 = Image.open(img)
    except:
        print('IMAGE NOT FOUND')
        return 'IMAGE NOT FOUND'

    image0 = np.array(im0)
    image1 = np.array(im1)

    # make mask of where the transparent bits are
    if len(image0[0][0]) == 4:
        trans_mask = image0[:, :, 3] == 0
        image0[trans_mask] = [0, 0, 0, 255]

    if len(image1[0][0]) == 4:
        trans_mask = image1[:, :, 3] == 0
        image1[trans_mask] = [0, 0, 0, 255]

    image0 = cv2.cvtColor(image0, cv2.COLOR_RGBA2GRAY)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGBA2GRAY)

    plt.imshow(image0)
    plt.imshow(image1)
    plt.show()

    kp0, des0 = sift.detectAndCompute(image0, None)
    kp1, des1 = sift.detectAndCompute(image1, None)
    if len(kp0) == 0 and len(kp1) != 0:
        print('solid color asset')
        return 'solid color asset'
    elif len(kp0) != 0 and len(kp1) == 0:
        print('solid color image')
        return 'solid color image'
    elif len(kp1) == 0 and len(kp1) == 0:
        print('solid color asset and image')
        return 'solid color asset and image'


    # Add padding to image
    if image1.shape[0] < image0.shape[0] and image1.shape[1] < image0.shape[1]:
        vertical_padding = int((image0.shape[0] - image1.shape[0]) / 2)
        horizontal_padding = int((image0.shape[1] - image1.shape[1]) / 2)
        image1 = cv2.copyMakeBorder(image1, vertical_padding, vertical_padding, 0, 0, cv2.BORDER_CONSTANT)
        image1 = cv2.copyMakeBorder(image1, 0, 0, horizontal_padding, horizontal_padding, cv2.BORDER_CONSTANT)
    else:
        image0 = cv2.copyMakeBorder(image0, 50, 50, 0, 0, cv2.BORDER_CONSTANT)
        image0 = cv2.copyMakeBorder(image0, 0, 0, 50, 50, cv2.BORDER_CONSTANT)
        image1 = cv2.copyMakeBorder(image1, 50, 50, 0, 0, cv2.BORDER_CONSTANT)
        image1 = cv2.copyMakeBorder(image1, 0, 0, 50, 50, cv2.BORDER_CONSTANT)

    # Multiple variants matching
    l_image1 = []
    image1_hor_flip = cv2.flip(image1, 1)
    image1_ver_flip = cv2.flip(image1, 0)
    l_image1.append(image1)
    l_image1.append(image1_hor_flip)
    l_image1.append(image1_ver_flip)

    max_mpts0 = []
    for img in l_image1:
        # Compute SIFT keypoints and descriptors
        kp0, des0 = sift.detectAndCompute(image0, None)
        kp1, des1 = sift.detectAndCompute(img, None)

        mpts0 = []

        # Matching descriptor using KNN algorithm
        if des1 is None or des0 is None:
            print('MATCH NOT FOUND')
            return 'MATCH NOT FOUND'
        elif len(des1) == 1 or len(des0) == 1:
            matches = flann.knnMatch(des1, des0, k=1)
        else:
            matches = flann.knnMatch(des1, des0, k=2)

        if len(matches) == 0:
            continue

        # Find good matches
        for match in matches:
            if len(match) != 1:
                if match[0].distance < uniqueness * match[1].distance:
                    mpts0.append(kp0[match[0].trainIdx].pt)
            else:
                mpts0.append(kp0[match[0].trainIdx].pt)

        if len(mpts0) != 0:
            if len(mpts0) > len(max_mpts0):
                max_mpts0 = mpts0

    # Discard outliers
    for i in range(num_iterations):
        max_mpts0, single_point = discard_outliers(outlier_threshold, max_mpts0)

    # Calculate image coordinates in asset
    if len(max_mpts0) > 0:
        mean_p = np.mean(max_mpts0, axis=0)
        print("Suggested coordinates = ", (mean_p[0], mean_p[1]))

        # Plot match
        if show_match:
            plot_image_pair([image0, image1])
            plot_matches(False, max_mpts0, single_point)
        return mean_p

    print('MATCH NOT FOUND')
    return 'MATCH NOT FOUND'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PICMATCHING')
    parser.add_argument('-asset', default=None)
    parser.add_argument('-img', default=None)
    parser.add_argument('-batch', default=None)
    parser.add_argument('-show_match', default=0)
    parser.add_argument('-uniqueness', default=0.65)
    parser.add_argument('-outlier_threshold', default=0)
    parser.add_argument('-num_iterations', default=3)
    parser.add_argument('-checkpoint',default=50)
    args = parser.parse_args()

    print('---------------MATCH START---------------')
    start_time = time.time()
    if args.batch == None:
        feature_matching(float(args.uniqueness), float(args.outlier_threshold), int(args.num_iterations), args.asset, args.img, int(args.show_match))
    else:
        file = pd.read_csv(args.batch)
        matches = pd.DataFrame({'asset': [],'img': [], 'coordinates': []}, dtype=object)
        for i in range(len(file)):
            prediction = feature_matching(args.uniqueness, args.outlier_threshold, args.num_iterations, file.iloc[i,0], file.iloc[i,1], False)
            matches.loc[len(matches.index)] = [file.iloc[i,0], file.iloc[i,1], prediction]

            if (i+1) % int(args.checkpoint) == 0:
                matches.to_csv("results.csv", index=False)

        matches.to_csv("results.csv", index=False)
    print('---------------MATCH END---------------')
    print('Total matching time = %.2fs' % (time.time()-start_time))
