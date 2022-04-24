import torch
import numpy as np


def iou(pred, target):
    N, num_class, h, w = pred.shape
    iou = np.zeros((N, num_class))
    for i in range(N):
        pred_array = torch.argmax(pred[i], dim=0).cpu().detach().numpy()
        target_array = torch.argmax(target[i], dim=0).cpu().numpy()
        for j in range(num_class):
            intersection = len(np.where(target_array[np.where(pred_array==j)]==j)[0])
            union = len(np.where(pred_array==j)[0]) + len(np.where(target_array==j)[0]) - intersection
            iou[i,j] += intersection/(union+0.00001)

    return iou

    