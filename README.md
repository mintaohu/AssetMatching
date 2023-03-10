## Introduction
This repo contains a pretrained SuperPoint network and a pretrained SuperGlue network. The combination of two neural networks is used to do feature matching between an image and a game asset. Prediction about whether this image belongs to the game asset will be made based on the matching result.

SuperPoint here operates as the "front-end," detecting interest points and computing their accompanying descriptors.  For more details please see:
* Full paper PDF: [SuperPoint: Self-Supervised Interest Point Detection and Description](https://arxiv.org/abs/1712.07629)

SuperGlue here operates as the "middle-end," performing context aggregation, matching, and filtering in a single end-to-end architecture. For more details, please see:
* Full paper PDF: [SuperGlue: Learning Feature Matching with Graph Neural Networks](https://arxiv.org/abs/1911.11763).

Two thresholds are used to operate as the "back-end", making prediction based on the results from superglue.

## Dependencies
* Python 3 >= 3.10
* PyTorch >= 1.12
* OpenCV >= 4.6
* Matplotlib >= 3.6
* NumPy >= 1.23


## Matching
```
python test.py -asset ASSET_PATH -img IMG_PATH
```

## Evaluation Results

Recall = 92.1% (based on 7500 pos instances)
<p align="center">
  <img src="assets/recall_result.png" width="400">
</p>

Specificity = 98.8% (based on 7500 neg instances)
<p align="center">
  <img src="assets/specificity_result.png" width="400">
</p>

Approximated accuracy  ≈ 97.1% (based on 7500 pos instances and 22500 neg instances)


## Example matches on real cases
<p align="center">
  <img src="assets/1.png" width="1000">
</p>
<p align="center">
  <img src="assets/2.png" width="1000">
</p>
<p align="center">
  <img src="assets/3.png" width="1000">
</p>


