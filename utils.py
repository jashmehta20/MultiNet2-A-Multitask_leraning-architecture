import torch
import numpy as np


def iou(pred, target):
    N, num_class, h, w = pred.shape
    iou=[0]*num_class
    for i in range(N):
        pred = torch.argmax(pred[i], dim=0).cpu().detach().numpy()
        target = torch.argmax(target[i], dim=0).cpu().numpy()
        for j in range(num_class):
            intersection = len(np.where(target[np.where(pred==j)]==j))
            union = len(np.all(pred==j)) + len(np.all(target==j)) - intersection
            iou[j] += intersection/union

    