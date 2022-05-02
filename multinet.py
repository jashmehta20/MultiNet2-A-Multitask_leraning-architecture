import torch.nn as nn
from fcn8_decoders import SegmentationDecoderFCN8s, DepthDecoderFCN8s, NormalDecoderFCN8s


class MultiNetFCN8(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super(MultiNetFCN8, self).__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.seg_block = SegmentationDecoderFCN8s(n_class=n_class)
        self.depth_block = DepthDecoderFCN8s()
        self.normal_block = NormalDecoderFCN8s()

    def forward(self, x):
        x = self.pretrained_net(x)
        seg_out = self.seg_block(x)
        depth_out = self.depth_block(x)
        normal_out = self.normal_block(x)

        return seg_out, depth_out, normal_out  # size=(N, n_class, x.H/1, x.W/1)
