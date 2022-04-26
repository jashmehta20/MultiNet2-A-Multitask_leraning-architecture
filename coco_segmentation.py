# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:50:28 2022

@author: acer
"""

from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset
from torchvision import io
import numpy as np
from typing import List, Tuple

class COCOsegmentation(Dataset):
    
    def __init__(self, annotation=COCO, train_path=None, img_id=List[int], ann_id=List[int], transform=None):
        
        self.train_path = train_path
        self.annotation = annotation
        self.img_data = annotation.loadImgs(img_id)
        self.ann_id = ann_id
        self.folder = [str(train_path/pic["file_name"]) for pic in self.img_data]
        self.transform = transform
        
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        
        ann_ids = self.annotation.getAnnIds(imgIds=self.img_data[i]['id'], catids=self.cat_ids)
        anns = self.annotations.loadAnns(ann_ids)
        mask = (np.max(np.stack([self.annotations.annToMask(ann) * ann["category_id"] for ann in anns]), axis=0)).unsqueeze(0)
        mask = torch.LongTensor(mask)
        
        pic = io.read_image(self.folder[i])
        ind = pic.shape[0]
        if ind == 1:
            pic = torch.cat([pic]*3)
        
        if self.transform is not None:
            return self.transform(pic, mask)
        
        return pic, mask
    
    def __len__(self) -> int:
        return len(self.folder)