from turtle import forward
import torch
from torch import nn

class ClassificationLoss(nn.Module):

    def __init__(self) -> None:
        super(ClassificationLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, y_hat, y):
        return self.loss(y_hat,y)


class SegmentationLoss(nn.Module):

    def __init__(self) -> None:
        super(SegmentationLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, y_hat, y):
        return self.loss(y_hat,y)


class DetectionLoss(nn.Module):

    def __init__(self) -> None:
        super(ClassificationLoss, self).__init__()
        pass

class MultinetLoss(nn.Module):

    def __init__(self) -> None:
        super(MultinetLoss, self).__init__()
        self.class_loss = ClassificationLoss()
        self.seg_loss = SegmentationLoss()

    def forward(self, seg, seg_hat, classification, classification_hat):
        seg_loss = self.seg_loss(seg_hat, seg)
        class_loss = self.class_loss(classification_hat, classification)
        return seg_loss+class_loss

