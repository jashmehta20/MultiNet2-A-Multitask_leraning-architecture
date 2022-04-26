# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 18:39:42 2022

@author: acer
"""

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
from typing import Tuple

def train_transform(pic_1: torch.LongTensor, pic_2: torch.LongTensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
    params = transforms.RandomResizedCrop.get_params(pic_1, scale=(0.5, 1.0), ratio=(0.75, 1.33))
    
    pic_1 = TF.resized_crop(pic_1, *params, size=(128,128))
    pic_2 = TF.resized_crop(pic_2, *params, size=(128,128))
    
    if random.random() > 0.5:
        pic_1 = TF.hflip(pic_1)
        pic_2 = TF.hflip(pic_2)
        
    return pic_1, pic_2