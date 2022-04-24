from torch import nn
from decoders import ClassificationDecoder, SegmentationDecoder, DetectionDecoder
from decoders.segmentation_decoder import FCN8s
from encoder.vgg16 import VGGNet

class Multinet(nn.Module):

    def __init__(self, num_classes) -> None:
        super(Multinet, self).__init__()

        self.encoder = VGGNet()
        self.seg_decoder = FCN8s(n_class=num_classes)


    def forward(self, x):

        x = self.encoder(x)
        seg_out = self.seg_decoder(x)

        return seg_out

