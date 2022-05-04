import torch
from encoders import VGGNet, ResNet101
from multinet import MultiNetFCN8
from torch import optim
from dataset import NYUv2
from multinet_utils import *
"""
eg: python train_multinet.py --dataroot /data --apply_augmentation --ckpt_dir /ckpt --epochs 200 --batch_size 4 --backbone resnet101 --architecture fcn
"""
import argparse

parser = argparse.ArgumentParser(description='Multi-task: Dense')
parser.add_argument('--dataroot', default='nyuv2', type=str, help='dataset root')
parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on NYUv2')
parser.add_argument('--ckpt_dir', default='/ckpt', type=str, help='checkpoint dir')
parser.add_argument('--epochs', default=200, type=int, help='no. of epochs')
parser.add_argument('--batch_size', default=4, type=int, help='batch_size')
parser.add_argument('--backbone', default='vgg16', type=str, help='vgg or resnet101')
parser.add_argument('--architecture', default='fcn8', type=str, help='fcn8 or fcn')
parser.add_argument('--resume', action='store_true', help='toggle to resume from checkpoint')
parser.add_argument('--ckpt_path', default='', type=str, help='path of checkpoint for resume')
opt = parser.parse_args()

if opt.backbone == 'resnet101':
    pretrained_model = ResNet101()

elif opt.backbone == 'vgg16':
    pretrained_model = VGGNet()

if opt.architecture == 'fcn8':
    model2 = MultiNetFCN8(pretrained_model, 13).cuda()

print("Model created")
if opt.resume:
    state_dict=torch.load(opt.ckpt_path)
    model2.load_state_dict(state_dict["model_state_dict"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
optimizer = optim.Adam(model2.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
nyuv2_test_set = NYUv2(root=opt.dataroot, train=False)

if opt.apply_augmentation:
    nyuv2_train_set = NYUv2(root=opt.dataroot, train=True, augmentation=True)

else:
    nyuv2_train_set = NYUv2(root=opt.dataroot, train=True, augmentation=False)

batch_size = opt.batch_size
nyuv2_train_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_train_set,
    batch_size=batch_size,
    shuffle=True)

nyuv2_test_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_test_set,
    batch_size=batch_size,
    shuffle=False)

print("dataset created")
print(f"No of epochs: {opt.epochs}")
print(f"Batch size: {opt.batch_size}")
print(f"Model Backbone: {opt.backbone}")
print(f"model parameters = {sum(i.numel() for i in model2.parameters())}")

multi_net_trainer(nyuv2_train_loader,
                    nyuv2_test_loader,
                    model2,
                    device,
                    optimizer,
                    scheduler,
                    opt,
                    n_class=13)
