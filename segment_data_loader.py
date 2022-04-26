# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 20:11:22 2022

@author: acer
"""
from coco_segmentation import COCOsegmentation
from utils_segmentation import train_transform
from torch.utils.data import DataLoader
from pycocotools.coco import COCO

BATCH_SIZE = 32

train_annot = COCO("F:/coco/annotations/instances_train2017.json")
valid_annot = COCO("F:/coco/annotations/instances_val2017.json")

cat_ids = train_annot.getCatIds(supNms=["person", "vehicle"])
train_img_ids = []
for cat in cat_ids:
    train_img_ids.extend(train_annot.getImgIds(catIds=cat))
    
train_img_ids = set(train_img_ids)
train_img_ids = list(train_img_ids)

valid_img_ids = []
for cat in cat_ids:
    valid_img_ids.extend(valid_annot.getImgIds(catIds=cat))
    
valid_img_ids = set(valid_img_ids)
valid_img_ids = list(valid_img_ids)
 
train_data = COCOsegmentation(train_annot, train_img_ids, cat_ids, "F:/coco/train2017", train_transform)
valid_data = COCOsegmentation(valid_annot, valid_img_ids, cat_ids, "F:/coco/val2017", train_transform)

valid_img_ids = []
for cat in cat_ids:
    valid_img_ids.extend(valid_annot.getImgIds(catIds=cat))
    
valid_img_ids = list(set(valid_img_ids))

train_load = DataLoader(
    train_data,
    BATCH_SIZE, 
    shuffle=True, 
    drop_last=True, 
    num_workers=4,
)

valid_load = DataLoader(
    valid_data,
    BATCH_SIZE, 
    shuffle=False, 
    drop_last=False, 
    num_workers=4,
)