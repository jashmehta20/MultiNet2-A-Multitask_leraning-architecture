from torch import nn

class SegmentationDecoder(nn.Module):

    def __init__(self, num_classes) -> None:
        super(SegmentationDecoder, self).__init__()

        self.conv512_2 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
        self.conv128_2 = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1)
        self.upscore2_1 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=256, kernel_size=2, stride=2, bias=False)
        self.upscore2_2 = nn.ConvTranspose2d(in_channels=256, out_channels=num_classes, kernel_size=2, stride=2, bias=False)
        self.upscore2_3 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=8, stride=8, bias=False)
        
    def forward(self, x1, dim_62_46, dim_31_23):

        x = self.conv512_2(x1)
        x = self.upscore2_1(x)
        x = x + dim_31_23
        x = self.upscore2_2(x)
        x3 = self.conv128_2(dim_62_46)
        x = x + x3  
        x = self.upscore2_3(x)
        return x
