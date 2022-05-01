import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationDecoderFCNs(nn.Module):

    def __init__(self, n_class) -> None:
        super(SegmentationDecoderFCNs, self).__init__()
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        x5 = x['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = x['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = x['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = x['x2']
        x1 = x['x1']
        score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))            # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)                      # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.relu(self.deconv3(score))
        score = self.bn3(score + x2)                      # size=(N, 128, x.H/4, x.W/4)
        score = self.relu(self.deconv4(score))
        score = self.bn4(score + x1)                      # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)
        score = F.log_softmax(score, dim=1)                   

        return score  # size=(N, n_class, x.H/1, x.W/1)

class DepthDecoderFCNs(nn.Module):

    def __init__(self) -> None:
        super(DepthDecoderFCNs, self).__init__()
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x5 = x['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = x['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = x['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = x['x2']
        x1 = x['x1']
        score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))            # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)                      # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.relu(self.deconv3(score))
        score = self.bn3(score + x2)                      # size=(N, 128, x.H/4, x.W/4)
        score = self.relu(self.deconv4(score))
        score = self.bn4(score + x1)                      # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)
        return score  # size=(N, n_class, x.H/1, x.W/1)

class NormalDecoderFCNs(nn.Module):

    def __init__(self) -> None:
        super(NormalDecoderFCNs, self).__init__()
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x):
        x5 = x['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = x['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = x['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = x['x2']
        x1 = x['x1']
        score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))            # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)                      # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.relu(self.deconv3(score))
        score = self.bn3(score + x2)                      # size=(N, 128, x.H/4, x.W/4)
        score = self.relu(self.deconv4(score))
        score = self.bn4(score + x1)                      # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)
        return score  # size=(N, n_class, x.H/1, x.W/1)