import torch
from encoders import VGGNet
from fcn8_decoders import SegmentationDecoderFCN8s, DepthDecoderFCN8s, NormalDecoderFCN8s
from torch import optim
from dataset import NYUv2
from utils import *

import argparse

parser = argparse.ArgumentParser(description='Multi-task: Dense')
parser.add_argument('--dataroot', default='nyuv2', type=str, help='dataset root')
parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on NYUv2')
parser.add_argument('--ckpt_dir', default='/ckpt', type=str, help='checkpoint dir')
parser.add_argument('--epochs', default=200, type=int, help='no. of epochs')
parser.add_argument('--batch_size', default=4, type=int, help='batch_size')
opt = parser.parse_args()

model = VGGNet()
model2 = NormalDecoderFCN8s(model).cuda()
# state_dict=torch.load(r"C:\Users\neetm\Desktop\DL\project\mtan\results\fcn_depth\ckpt\model_epoch_182.pth")
# model2.load_state_dict(state_dict["model_state_dict"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
optimizer = optim.Adam(model2.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
nyuv2_test_set = NYUv2(root=opt.dataroot, train=False)

if opt.apply_augmentation:
    nyuv2_train_set = NYUv2(root=opt.dataroot, train=True, augmentation=True, small=790)

else:
    nyuv2_train_set = NYUv2(root=opt.dataroot, train=True, augmentation=False, small=646)

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
                    model2,
                    device,
                    optimizer,
                    scheduler,
                    'normal',
                    opt.ckpt_dir,
                    200,
                    n_class=1)
