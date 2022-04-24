from itertools import cycle
import torch
from loss.loss import MultinetLoss
from utils import iou as calc_iou
from multinet import Multinet
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from data import PascalVOC
import multinet
import numpy as np
import matplotlib.pyplot as plt
import torchmetrics.functional as F
from scipy.spatial.distance import dice
from torch.optim import lr_scheduler
device = 'cuda' if torch.cuda.is_available() else 'cpu'

LEARNING_RATE = 0.00001
BATCH_SIZE = 4
NUM_EPOCHS = 100
num_class = 21
data_split = 0.1
to_pil = transforms.ToPILImage()

def val():
    model.eval()
    iou = None
    for idx, (image, seg_label) in enumerate(val_loader):
        image, seg_label = image.to(device), seg_label.to(device)
        seg_pred = model(image)
        
        if iou is None:
            iou = calc_iou(seg_pred, seg_label)

        else:
            iou = np.concatenate((iou, calc_iou(seg_pred, seg_label)), axis=0)
    # visualize(seg_pred[0], seg_label[0])


    return np.mean(iou, axis=0)


def visualize(pred, target):
    
    sg = torch.nn.Sigmoid()
    bus_pred, bicycle_pred, car_pred = sg(pred[0]), sg(pred[1]), sg(pred[2])
    bus_pred[bus_pred>0.5]=1
    bus_pred[bus_pred<=0.5]=0
    bicycle_pred[bicycle_pred>0.5]=1
    bicycle_pred[bicycle_pred<=0.5]=0
    car_pred[car_pred>0.5]=1
    car_pred[car_pred<=0.5]=0

    bus_target, bicycle_target, car_target = target[0], target[1], target[2]
    prediction = torch.cat( (bus_pred, bicycle_pred, car_pred) , dim=1)
    ground_truth = torch.cat( (bus_target, bicycle_target, car_target) , dim=1)
    plt.imshow(torch.cat((prediction, ground_truth), dim=0))
    plt.title("compare")
    plt.show()

# image_transforms = transforms.Compose([transforms.Resize((375,500))])
train_dataset = PascalVOC(csv_loc="train.csv", img_resize=(512,512), num_classes=3)
train_len = int(len(train_dataset)*data_split) 
val_len = len(train_dataset)-train_len
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_len, val_len])
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)



model = Multinet(num_classes=num_class).to(device)
criteria = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
to_image = transforms.ToPILImage()

for epoch in range(NUM_EPOCHS):
    for idx, (image, seg_label) in enumerate(train_dataloader):

        image, seg_label = image.to(device), seg_label.to(device)
        seg_pred = model(image)
        loss = criteria(seg_pred, seg_label)
        loss.backward()
        optimizer.step()
        print(idx)
    print(f"iou = {val()} ---- epoch = {epoch}")

    if epoch%10==0:
        torch.save({"epoch":epoch, "model_state_dict":model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), 'loss':loss}, f"checkpoints\\multinet_epoch_{epoch}.pth")

