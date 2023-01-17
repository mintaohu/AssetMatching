import torch

from superpoint import SuperPoint
from superglue import SuperGlue


class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """
    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.superglue = SuperGlue(config.get('superglue', {}))

    def forward(self, data):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}
        skip = False

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if 'keypoints0' not in data:
            try:
                pred0 = self.superpoint({'image': data['image0']})
                pred = {**pred, **{k + '0': v for k, v in pred0.items()}}
            except:
                skip = True

        if 'keypoints1' not in data:
            try:
                pred1 = self.superpoint({'image': data['image1']})
                pred = {**pred, **{k + '1': v for k, v in pred1.items()}}
            except:
                skip = True

        if skip == True:
            return pred, skip

        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}

        descs1 = data['descriptors0'][0].cpu().numpy()
        # print(descs1.shape)
        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        try:
            pred = {**pred, **self.superglue(data)}
        except:
            skip = True

        return pred, skip
