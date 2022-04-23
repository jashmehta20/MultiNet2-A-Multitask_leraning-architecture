import torchvision.models as models
from torch import nn
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VGG16(nn.Module):

    def __init__(self) -> None:
        super(VGG16,self).__init__()

        self.vgg16_features = models.vgg16(pretrained=True).features
        self.conv_scale_1 = nn.Conv2d(256, 128, 1)
        self.conv_scale_2 = nn.Conv2d(512, 256, 1)
        

    def forward(self, x):
        encoder_output = self.vgg16_features(x)
        scale_1_output = self.conv_scale_1(self.vgg16_features[:17](x))
        scale_2_output = self.conv_scale_2(self.vgg16_features[:24](x))

        return (encoder_output, scale_1_output, scale_2_output)





