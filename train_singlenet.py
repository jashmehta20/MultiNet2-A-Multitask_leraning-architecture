import torch
from encoders import VGGNet, ResNet101
from fcn8_decoders import SegmentationDecoderFCN8s, DepthDecoderFCN8s, NormalDecoderFCN8s
from torch import optim
from dataset import NYUv2
from multinet_utils import *

import argparse

parser = argparse.ArgumentParser(description='Multi-task: Dense')
parser.add_argument('--dataroot', default='nyuv2', type=str, help='dataset root')
parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on NYUv2')
parser.add_argument('--ckpt_dir', default='/ckpt', type=str, help='checkpoint dir')
parser.add_argument('--epochs', default=200, type=int, help='no. of epochs')
parser.add_argument('--batch_size', default=4, type=int, help='batch_size')
parser.add_argument('--task', default='semantic', type=str, help='task')
parser.add_argument('--backbone', default='vgg16', type=str, help='vgg or resnet101')
parser.add_argument('--resume', action='store_true', help='toggle to resume from checkpoint')
parser.add_argument('--ckpt_path', default='', type=str, help='path of checkpoint for resume')
opt = parser.parse_args()

classes = 13

if opt.backbone == 'resnet101':
    pretrained_model = ResNet101()

elif opt.backbone == 'vgg16':
    pretrained_model = VGGNet()

if opt.task == 'semantic':
    classes = 13
    model = SegmentationDecoderFCN8s(pretrained_model, classes)

if opt.task == 'depth':
    classes=1
    model = DepthDecoderFCN8s(pretrained_model)

if opt.task == 'normal':
    classes = 3
    model = NormalDecoderFCN8s(pretrained_model)
if opt.resume:
    state_dict=torch.load(opt.ckpt_path)
    model.load_state_dict(state_dict["model_state_dict"])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
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

single_task_trainer(nyuv2_train_loader,
                    nyuv2_test_loader,
                    model,
                    device,
                    optimizer,
                    scheduler,
                    opt.task,
                    opt.ckpt_dir,
                    opt.epochs,
                    n_class=classes)
