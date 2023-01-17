import cv2
import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def feature_matching(asset, img, horizontal_flip_toggle):
    # Read in images from a filepath as graycsale.
    image0 = cv2.imread(asset, cv2.IMREAD_GRAYSCALE)
    image1 = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    if image1.shape[0] < image0.shape[0]:
        vertical_padding = int((image0.shape[0] - image1.shape[0]) / 2)
        image1 = cv2.copyMakeBorder(image1, vertical_padding, vertical_padding, 0, 0, cv2.BORDER_CONSTANT)

    if image1.shape[1] < image0.shape[1]:
        horizontal_padding = int((image0.shape[1] - image1.shape[1]) / 2)
        image1 = cv2.copyMakeBorder(image1, 0, 0, horizontal_padding, horizontal_padding, cv2.BORDER_CONSTANT)

    if horizontal_flip_toggle:
        image1 = cv2.flip(image1, 1)

    # Initiate SIFT detector
    orb = cv2.ORB_create()

    # Compute SIFT keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(image0, None)
    kp2, des2 = orb.detectAndCompute(image1, None)
    print(des1.shape)
    print(des2.shape)
    # FLANN parameters and initialize
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Matching descriptor using KNN algorithm
    if des1 is None or des2 is None:
        return 0
    elif len(des1) == 1 or len(des2) == 1:
        matches = flann.knnMatch(des1, des2, k=1)
    else:
        matches = flann.knnMatch(des1, des2, k=2)

    if len(matches[0]) == 1:
        return 0

    # Create a mask to draw all good matches
    matchesMask = []

    # Store all good matches as per Lowe's Ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append(m)
            matchesMask.append([0, 0])  # Match
        else:
            matchesMask.append([0, 0])  # Mismatch

    # Draw all good matches
    draw_params = dict(  # matchColor = (0,255,0),  #If you want a specific colour
        # singlePointColor = (255,0,0), #If you want a specific colour
        matchesMask=matchesMask,
        flags=cv2.DrawMatchesFlags_DEFAULT)

    good_matches = cv2.drawMatchesKnn(image0, kp1, image1, kp2, matches, None, **draw_params)

    plt.figure(figsize=(15, 15))

    plt.imshow(good_matches)
    plt.title('All good matches')
    plt.axis('off')

data = pd.read_csv("data/test/labels.csv")

# Ratio of positive to negative samples = 1:3
# neg_data = data.iloc[:273908,:]
# neg_data = neg_data.sample(frac=1)
# neg_data = neg_data.iloc[:22500,:]
#
# pos_data= data.iloc[273908:273908+7500,:]
#
# mixed_data = pd.concat([pos_data, neg_data])
# mixed_data = mixed_data.sample(frac=1)


# shuffle pos data
# pos_data= data.iloc[273908:273908+7500,:]
# pos_data = pos_data.sample(frac=1)

# shuffle neg data
# neg_data = data.iloc[:273908,:]
# neg_data = neg_data.sample(frac=1)
# neg_data = neg_data.iloc[:7500,:]



# shuffle data
# data = data.sample(frac=1)

img_root = "data/test/temp/"
asset_root = "data/test/img/"



# Single instance testing
plt.axis('off')
idx = 277976
prediction = feature_matching(os.path.join(asset_root, data.iloc[idx, 0]), os.path.join(img_root, data.iloc[idx, 1]), 0)

num_correct_predictions = 0
l_acc = []
l_batch_no = []

# Recall = (TP/TP+FN)
# for i in range(len(pos_data)):
#     prediction = feature_matching(os.path.join(asset_root, pos_data.iloc[i, 0]), os.path.join(img_root, pos_data.iloc[i, 1]), 0)
#     hf_prediction = feature_matching(os.path.join(asset_root, pos_data.iloc[i, 0]), os.path.join(img_root, pos_data.iloc[i, 1]), 1)
#
#     if prediction == int(pos_data.iloc[i, 2]) or hf_prediction == int(pos_data.iloc[i, 2]):
#         num_correct_predictions += 1
#
#     if (i+1) % 500 == 0:
#         batch_no = int((i+1) / 500)
#         l_batch_no.append(batch_no)
#         print("number of predictions: ", i+1)
#
#         accuracy = num_correct_predictions / (i+1)
#         l_acc.append(accuracy)
#         print("current accuracy is: ", accuracy)


# Specificity = TN/TN+FP
# for i in range(len(neg_data)):
#     prediction = feature_matching(os.path.join(asset_root, neg_data.iloc[i, 0]), os.path.join(img_root, neg_data.iloc[i, 1]), 0)
#     hf_prediction = feature_matching(os.path.join(asset_root, neg_data.iloc[i, 0]), os.path.join(img_root, neg_data.iloc[i, 1]), 1)
#
#     if prediction == int(neg_data.iloc[i, 2]) or hf_prediction == int(neg_data.iloc[i, 2]):
#         num_correct_predictions += 1
#
#     if (i+1) % 500 == 0:
#         batch_no = int((i+1)/500)
#         l_batch_no.append(batch_no)
#         print("number of predictions: ", i+1)
#         accuracy = num_correct_predictions/(i+1)
#         l_acc.append(accuracy)
#         print("current accuracy is: ", num_correct_predictions/(i+1))


# Accuracy = (TP+TN)/(TP+FP+TN+FN)
# for i in range(len(mixed_data)):
#     prediction = feature_matching(os.path.join(asset_root, mixed_data.iloc[i, 0]), os.path.join(img_root, mixed_data.iloc[i, 1]), 0)
#     hf_prediction = feature_matching(os.path.join(asset_root, mixed_data.iloc[i, 0]), os.path.join(img_root, mixed_data.iloc[i, 1]), 1)
#
#     if prediction == int(mixed_data.iloc[i, 2]) or hf_prediction == int(mixed_data.iloc[i, 2]):
#         num_correct_predictions += 1
#
#     if (i+1) % 2000 == 0:
#         batch_no = int((i+1)/2000)
#         l_batch_no.append(batch_no)
#         print("number of predictions: ", i+1)
#         accuracy = num_correct_predictions/(i+1)
#         l_acc.append(accuracy)
#         print("current accuracy is: ", accuracy)

# plot the figure
# plt.figure()
# plt.plot(l_batch_no, l_acc)
# plt.xticks(range(1, max(l_batch_no)+1))
# plt.show()












