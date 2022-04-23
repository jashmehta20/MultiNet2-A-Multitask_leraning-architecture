from torch import nn


class ClassificationDecoder(nn.Module):

    def __init__(self, in_channels, num_classes=2) -> None:
        super(ClassificationDecoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=30, kernel_size=3)
        self.classifier = nn.Linear(5880, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 5880)
        x = self.classifier(x)
        return x
