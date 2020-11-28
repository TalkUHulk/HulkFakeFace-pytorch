import torch.nn as nn
from torchvision.models import resnet50
import torch


class ResNet(nn.Module):
    expansion = 1

    def __init__(self, num_classes=2):
        super(ResNet, self).__init__()
        self.resnet50 = resnet50(num_classes=1000, pretrained=True)
        self.layer0 = nn.Sequential(self.resnet50.conv1,
                                    self.resnet50.bn1,
                                    self.resnet50.relu,
                                    self.resnet50.maxpool
                                    )
        self.layer1 = self.resnet50.layer1
        self.layer2 = self.resnet50.layer2
        self.layer3 = self.resnet50.layer3
        self.layer4 = self.resnet50.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        feature = torch.flatten(x, 1)
        x = self.fc(feature)
        return feature, x

# from loss.center_loss import CenterLoss
# loss = CenterLoss(num_classes=2, feat_dim=2048, use_gpu=False)
# a = torch.rand([3, 3, 112, 112])
# model = ResNet()
# c, b = model(a)
# print(b.shape, c.shape)
# l = loss(c.squeeze(), torch.ones([3]))
# print(l)

# l = torch.tensor([0, 1, 1, 0])
# label = torch.nn.functional.one_hot(l, 2)
# print(label)
