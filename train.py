import torch
from loss.loss import MultinetLoss

from multinet import Multinet
from encoder import VGG16
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from data import PascalVOC
import multinet
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

LEARNING_RATE = 0.0001
BATCH_SIZE = 4
NUM_EPOCHS = 5

image_transforms = transforms.Compose([transforms.Resize((375,500))])
train_dataset = PascalVOC(csv_loc="train.csv", img_resize=(512,512), num_classes=3)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)



model = Multinet(num_classes=3).to(device)
criteria = MultinetLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

to_image = transforms.ToPILImage()

for epoch in range(NUM_EPOCHS):
    for idx, (image, seg_label, class_label) in enumerate(train_dataloader):

        image, seg_label, class_label = image.to(device), seg_label.to(device), class_label.to(device)
        seg_pred, class_pred = model(image)
        loss = criteria(seg_label, seg_pred, class_label, class_pred)
        loss.backward()
        optimizer.step()
        
    
