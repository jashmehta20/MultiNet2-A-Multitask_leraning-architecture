from torch import nn

class SegmentationDecoder(nn.Module):

    def __init__(self) -> None:
        super(SegmentationDecoder, self).__init__()

        self.conv512_2 = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=1)
        self.conv128_2 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)
        self.upscore2_1 = nn.ConvTranspose2d(in_channels=2, out_channels=256, kernel_size=2, stride=2, bias=False)
        self.upscore2_2 = nn.ConvTranspose2d(in_channels=256, out_channels=2, kernel_size=2, stride=2, bias=False)
        self.upscore2_3 = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=8, stride=8, bias=False)
        
    def forward(self, x1, dim_156_48, dim_78_24):

        x = self.conv512_2(x1)
        x = self.upscore2_1(x)
        x = x + dim_78_24
        x = self.upscore2_2(x)
        x3 = self.conv128_2(dim_156_48)
        x = x + x3
        x = self.upscore2_3(x)
        return x
