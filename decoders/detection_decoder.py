from torch import nn

class DetectionDecoder(nn.Module):

    def __init__(self, in_channel=512) -> None:
        super(DetectionDecoder, self).__init__()  
        self.conv_1 = nn.Conv2d(in_channels=in_channel, out_channels=500, kernel_size=3)
        self.conv_layer = nn.Conv2d(in_channels=500, out_channels=6, kernel_size=(1,1))
        
        
    def forward(self, x):
        
        bott = self.conv_1(x)
        pred = self.conv_layer(bott)
        return pred
