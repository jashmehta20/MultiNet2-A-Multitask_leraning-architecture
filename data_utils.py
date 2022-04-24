import os
import pandas as pd
import numpy as np


DEETECTION_DIR = r"C:\Users\neetm\Desktop\DL\pascal\VOCdevkit\VOC2012\Annotations"
CLASSIFICATION_DIR = r"C:\Users\neetm\Desktop\DL\pascal\VOCdevkit\VOC2012\ImageSets\Main"
SEGMENTATION_DIR = r"C:\Users\neetm\Desktop\DL\pascal\VOCdevkit\VOC2012\SegmentationClass"
IMAGE_DIR = r"C:\Users\neetm\Desktop\DL\pascal\VOCdevkit\VOC2012\JPEGImages"
SEG_LIST = os.listdir(SEGMENTATION_DIR)
DET_LIST = os.listdir(DEETECTION_DIR)

def get_intersection(img_name):
    
    img_name, idx = img_name.split()
    classification = idx
    if classification=="1": cls=True 
    else: cls=False
    seg = img_name+".png" in  SEG_LIST
    det = img_name+".xml" in  DET_LIST
    return cls and det and seg

classes = {"bus":1, "bicycle":2, "car":3}
bicycle = [0,128,0]
car = [128,128,128]
bus = [128,128,0]
codes = {"bus":bus, "bicycle":bicycle, "car":car}


if __name__ == "__main__":
    df = pd.DataFrame(columns=["image_path", "seg_path", "classification", "class_name", "codes"])

    for object_class, idx in classes.items():
        class_path = os.path.join(CLASSIFICATION_DIR, f"{object_class}_train.txt")
        with open(class_path) as f:
            image_list = f.readlines()
            for img in image_list:
                truth = get_intersection(img)
                img = img.split()
                img = img[0]
                if truth:
                    row = [os.path.join(IMAGE_DIR, img + ".jpg"), os.path.join(SEGMENTATION_DIR, img + ".png"),
                    idx, object_class, codes[object_class]]
                    df.loc[len(df.index)] = row

    df.to_csv("train.csv")
            
    

        
