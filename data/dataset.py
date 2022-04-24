from sklearn import datasets
import torch
import torchvision.transforms.transforms as transforms
import cv2
import os
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
import cv2
from PIL import Image


classes = {"bus":0, "bicycle":1, "car":2, "background":3}
bicycle = [0,128,0]
car = [128,128,128]
bus = [0,128,128]
background = [0,0,0]
codes = {"bus":bus, "bicycle":bicycle, "car":car, 'background':background}

class PascalVOC(Dataset):

    def __init__(self, csv_loc, img_resize, num_classes, transform=None) -> None:
        super(PascalVOC, self).__init__()
        self.df = pd.read_csv(csv_loc)
        self.transform = transform
        self.resize = img_resize
        self.n_class = num_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = Image.open(self.df["image_path"][index])
        image = image.convert("RGB")
        image = image.resize(self.resize)
        seg_label = Image.open(self.df["seg_path"][index])
        seg_label = seg_label.convert("RGB")
        seg_label = seg_label.resize(self.resize)
        seg_label = np.array(seg_label)
        channel_list = []
        
        for img_class, code in codes.items():
            seg_target = np.zeros((512,512))
            seg_target[np.where(np.all(seg_label==code, axis=-1))]=1
            channel_list.append(seg_target)

        seg_target = np.reshape(np.concatenate((channel_list[0],channel_list[1], channel_list[2], channel_list[3])), (-1,512,512))

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = preprocess(image)
        seg_target = torch.from_numpy(seg_target)
        if self.transform is not None:
            image = self.transform(image)
            seg_target = self.transform(seg_target)
        
        return image, seg_target
        