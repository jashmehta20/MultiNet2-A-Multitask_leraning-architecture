import os
import pandas as pd
import numpy as np
from os.path import join as osp

DEETECTION_DIR = r"C:\Users\neetm\Desktop\DL\pascal\VOCdevkit\VOC2012\Annotations"
CLASSIFICATION_DIR = r"C:\Users\neetm\Desktop\DL\pascal\VOCdevkit\VOC2012\ImageSets\Main"
SEGMENTATION_DIR = r"C:\Users\neetm\Desktop\DL\pascal\VOCdevkit\VOC2012\SegmentationClass"
IMAGE_DIR = r"C:\Users\neetm\Desktop\DL\pascal\VOCdevkit\VOC2012\JPEGImages"
SEG_LIST = os.listdir(SEGMENTATION_DIR)
DET_LIST = os.listdir(DEETECTION_DIR)

# classes = {"bus":1, "bicycle":2, "car":3}
# bicycle = [0,128,0]
# car = [128,128,128]
# bus = [128,128,0]
# codes = {"bus":bus, "bicycle":bicycle, "car":car}
classes = {}

codes = {
               "background":[0, 0, 0],  # background
               "aeroplane":[128, 0, 0], # aeroplane
               "bicycle":[0, 128, 0], # bicycle
               "bird":[128, 128, 0], # bird
               "boat":[0, 0, 128], # boat
               "bottle":[128, 0, 128], # bottle
               "bus":[0, 128, 128], # bus 
               "car":[128, 128, 128], # car
               "cat":[64, 0, 0], # cat
               "chair":[192, 0, 0], # chair
               "cow":[64, 128, 0], # cow
               "diningtable":[192, 128, 0], # dining table
               "dog":[64, 0, 128], # dog
               "horse":[192, 0, 128], # horse
               "motorbike":[64, 128, 128], # motorbike
               "person":[192, 128, 128], # person
               "pottedplant":[0, 64, 0], # potted plant
               "sheep":[128, 64, 0], # sheep
               "sofa":[0, 192, 0], # sofa
               "train":[128, 192, 0], # train
               "tvmonitor":[0, 64, 128] # tv/monitor
}
k=0
for i,j in codes.items():
    j.reverse()
    codes[i] = j
    classes[i]=k
    k+=1



if __name__ == "__main__":
    df = pd.DataFrame(columns=["image_path", "seg_path"])
    for i in SEG_LIST:
        seg_path = osp(SEGMENTATION_DIR, i)
        image_path = osp(IMAGE_DIR, i.replace("png","jpg"))
        df.loc[len(df)] = [image_path, seg_path]

    df.to_csv('train.csv')