from itertools import cycle
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
import torchmetrics.functional as F
from scipy.spatial.distance import dice
device = 'cuda' if torch.cuda.is_available() else 'cpu'

LEARNING_RATE = 0.000001
BATCH_SIZE = 2
NUM_EPOCHS = 1
to_pil = transforms.ToPILImage()

def val():
    model.eval()
    dice_score = []
    for idx, (image, seg_label, class_label) in enumerate(val_loader):
        image, seg_label, class_label = image.to(device), seg_label.to(device), class_label.to(device)
        seg_pred, class_pred = model(image)
        seg_pred = seg_pred.cpu().detach()
        seg_label = seg_label.cpu()
        dice_score.append(iou(seg_pred, seg_label))
    # visualize(seg_pred[0], seg_label[0])


    return sum(dice_score)/len(dice_score)

def iou(pred, target):
    pred = pred.flatten()
    target = target.flatten()
    return dice(pred, target)

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
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [191, 12])
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)



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

    print(f"dice_score = {val()} epoch = {epoch}")

    if epoch%10==0:
        torch.save({"epoch":epoch, "model_state_dict":model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), 'loss':loss}, f"checkpoints\\multinet_epoch_{epoch}.pth")

