
import torch.nn as nn
from torchvision.models.vgg import VGG
import torchvision.models as models

class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output


ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class ResNet101(nn.Module):

    def __init__(self, pretrained=True) -> None:
        super(ResNet101, self).__init__()
        self.resnet101 = models.resnet101(pretrained)
        self.conv1_1x1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv2_1x1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv3_1x1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.conv4_1x1 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1)
        del self.resnet101._modules['avgpool']
        del self.resnet101._modules['fc']

    def forward(self, x):
        output = {}
        output['x1'] = self.resnet101._modules['conv1'](x)
        output['x2'] = self.resnet101._modules['maxpool'](self.resnet101._modules['relu'](self.resnet101._modules['bn1'](output['x1'])))
        x2 = self.conv1_1x1(output['x2'])
        output['x3'] = self.resnet101._modules['layer2'](self.resnet101._modules['layer1'](output['x2']))
        x3 = self.conv2_1x1(output['x3'])
        output['x4'] = self.resnet101._modules['layer3'](output['x3'])
        x4 = self.conv3_1x1(output['x4'])
        output['x5'] = self.resnet101._modules['layer4'](output['x4'])
        x5 = self.conv4_1x1(output['x5'])
        output['x2'] = x2
        output['x3'] = x3
        output['x4'] = x4
        output['x5'] = x5
        return output

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

