from torch import nn,cat
from torchvision import ops

class DetectionDecoder(nn.Module):

    def __init__(self, in_channel) -> None:
        super(DetectionDecoder, self).__init__()
        self.roialign = ops.roi_align()   
        self.conv_1 = nn.Conv2d(in_channels=in_channel, out_channels=500, kernel_size=3)
        self.conv_layer = nn.Conv2d(in_channels=512, out_channels=6, kernel_size=(1,1))
        
        
    def forward(self, x):
        
        bott = self.conv_1(x)
        pred = self.conv_layer(bott)
        concat = cat((self.roialign, self.conv_1, self.pred), 3)
        delta_pred = self.conv_layer(concat)
        return pred, delta_pred
