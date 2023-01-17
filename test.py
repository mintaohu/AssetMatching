import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import math

from matching import Matching
from utils import read_image_test, make_matching_plot, frame2tensor

# Predict whether image is in asset or not
def test(asset_path, img_path, device, matching):
    image0, image1 = read_image_test(asset_path, img_path, False)
    for i in range(len(image1)):
        for j in range(len(image0)):
            torch.cuda.empty_cache()
            inp0 = frame2tensor(image0[j], device)
            inp1 = frame2tensor(image1[i], device)
            pred, skip = matching({'image0': inp0, 'image1': inp1})
            if skip == True:
                continue

            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts1 = pred['keypoints1']
            matches = pred['matches0']
            valid = matches > -1
            if kpts1.shape[0] < 15:
                if matches[valid].shape[0] >= max(4, int(math.floor(kpts1.shape[0]) * 0.7)):
                    return 1
            else:
                if matches[valid].shape[0] >= max(4, int(math.floor(kpts1.shape[0]) / 3)):
                    return 1

    image0, image1 = read_image_test(asset_path, img_path, True)
    for i in range(len(image1)):
        for j in range(len(image0)):
            torch.cuda.empty_cache()
            inp0 = frame2tensor(image0[j], device)
            inp1 = frame2tensor(image1[i], device)
            pred, skip = matching({'image0': inp0, 'image1': inp1})
            if skip == True:
                continue
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts1 = pred['keypoints1']
            matches = pred['matches0']
            valid = matches > -1

            if kpts1.shape[0] < 15:
                if matches[valid].shape[0] >= max(4, int(math.floor(kpts1.shape[0]) * 0.7)):
                    return 1
            else:
                if matches[valid].shape[0] >= max(4, int(math.floor(kpts1.shape[0]) / 3)):
                    return 1

    return 0

# Visualize matching result
def visualize_test(asset_path, img_path, device, matching):
    image0, image1 = read_image_test(asset_path, img_path, False)
    image0, image1 = read_image_test(asset_path, img_path, True)

    for i in range(len(image1)):
        for j in range(len(image0)):
            torch.cuda.empty_cache()
            inp0 = frame2tensor(image0[j], device)
            inp1 = frame2tensor(image1[i], device)
            pred, skip = matching({'image0': inp0, 'image1': inp1})

            if skip == True:
                continue
            # Visualize the matches.
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']

            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            color = 'r'

            if kpts1.shape[0] < 15:
                if matches[valid].shape[0] >= max(4, int(math.floor(kpts1.shape[0]) * 0.7)):
                    make_matching_plot(image0[j], image1[i], kpts0, kpts1, mkpts0, mkpts1, color, True)
            else:
                if matches[valid].shape[0] >= max(4, int(math.floor(kpts1.shape[0]) / 3)):
                    make_matching_plot(image0[j], image1[i], kpts0, kpts1, mkpts0, mkpts1, color, True)

# config for superpoint and superglue
config = {
        'superpoint': {
            'nms_radius':  2,
            'keypoint_threshold': 0.005,
            'max_keypoints': 15000
        },
        'superglue': {
            'weights': 'picmatch',
            'sinkhorn_iterations': 100,
            'match_threshold': 0.4,
        }
    }

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_grad_enabled(False)
matching = Matching(config).eval().to(device)


# Test single instance
# asset_path = 'data/test/img/normal_pack_199.png'
# img_path = 'data/test/temp/26332.png'
# prediction = visualize_test(asset_path, img_path, device,matching)

# print('finish')

# Read annotation file
data = pd.read_csv("data/test/labels.csv")

# shuffle pos data
pos_data= data.iloc[273908:273908+7500,:]
pos_data = pos_data.sample(frac=1)

# shuffle neg data
neg_data = data.iloc[:273908,:]
neg_data = neg_data.sample(frac=1)
neg_data = neg_data.iloc[:7500,:]

img_root = "data/test/temp/"
asset_root = "data/test/img/"


# Recall = (TP/TP+FN)
num_correct_predictions = 0
l_batch_no = []
l_acc = []

for i in range(len(pos_data)):
    asset_path = os.path.join(asset_root, pos_data.iloc[i, 0])
    img_path = os.path.join(img_root, pos_data.iloc[i, 1])
    prediction = test(asset_path, img_path, device,matching)

    if prediction == int(pos_data.iloc[i, 2]):
        num_correct_predictions += 1
    print('pos_idx:', i+1)
    print('acc:', num_correct_predictions / (i + 1))

    if (i+1) % 500 == 0:
        batch_no = int((i+1) / 500)
        l_batch_no.append(batch_no)
        print("number of predictions: ", i+1)

        accuracy = num_correct_predictions / (i+1)
        l_acc.append(accuracy)
        print("current accuracy is: ", accuracy)

# plot the figure
plt.figure()
plt.plot(l_batch_no, l_acc)
plt.xticks(range(1, max(l_batch_no)+1))
plt.savefig('recall_result.png')


# Specificity = TN/TN+FP
num_correct_predictions = 0
l_batch_no = []
l_acc = []

for i in range(len(neg_data)):
    asset_path = os.path.join(asset_root, neg_data.iloc[i, 0])
    img_path = os.path.join(img_root, neg_data.iloc[i, 1])
    prediction = test(asset_path, img_path, device,matching)

    if prediction == int(neg_data.iloc[i, 2]):
            num_correct_predictions += 1
    print('neg_idx:', i+1)
    print('acc:', num_correct_predictions/(i+1))

    if (i+1) % 500 == 0:
        batch_no = int((i+1)/500)
        l_batch_no.append(batch_no)
        print("number of predictions: ", i+1)
        accuracy = num_correct_predictions/(i+1)
        l_acc.append(accuracy)
        print("current accuracy is: ", num_correct_predictions/(i+1))
        
# plot the figure
plt.figure()
plt.plot(l_batch_no, l_acc)
plt.xticks(range(1, max(l_batch_no)+1))
plt.savefig('specificity_result.png')
