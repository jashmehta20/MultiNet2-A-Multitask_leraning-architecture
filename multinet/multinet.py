from torch import nn
from decoders import ClassificationDecoder, SegmentationDecoder
from encoder import VGG16

class Multinet(nn.Module):

    def __init__(self) -> None:
        super(Multinet, self).__init__()

        self.encoder = VGG16()
        self.seg_decoder = SegmentationDecoder()
        self.classification_decoder = ClassificationDecoder(in_channels=512, num_classes=2)

    def forward(self, x):

        encoder_output, scale_1_output, scale_2_output = self.encoder(x)
        seg_output = self.seg_decoder(encoder_output, scale_1_output, scale_2_output)
        class_output = self.classification_decoder(encoder_output)

        return seg_output, class_output

