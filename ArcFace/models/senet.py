from abc import ABC
import torch.nn as nn
import torch

__all__ = ["se_resnet50_ir"]


def l2_norm(x, axis=1):
    norm = torch.norm(x, 2, axis, True)
    output = torch.div(x, norm)
    return output


class SEModule(nn.Module, ABC):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR_SE(nn.Module, ABC):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class IRSeResNet50(nn.Module, ABC):
    def __init__(self, block, layers, dropout_p=0.2):
        super(IRSeResNet50, self).__init__()

        self.layer0 = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.PReLU(64)
                                    )

        self.layer1 = self._make_layer(
            block,
            planes=64,
            expansion=1,
            blocks=layers[0],

        )
        self.layer2 = self._make_layer(
            block,
            planes=64,
            expansion=2,
            blocks=layers[1],

        )
        self.layer3 = self._make_layer(
            block,
            planes=128,
            expansion=2,
            blocks=layers[2],

        )
        self.layer4 = self._make_layer(
            block,
            planes=256,
            expansion=2,
            blocks=layers[3],

        )

        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(dropout_p),
                                          nn.Flatten(),
                                          nn.Linear(512 * 7 * 7, 512),
                                          nn.BatchNorm1d(512))

    def _make_layer(self, block, planes, expansion, blocks, stride=2):
        layers = []
        depth = planes * expansion
        layers.append(block(planes, depth, stride))
        for i in range(1, blocks):
            layers.append(block(depth, depth, 1))
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.output_layer(x)
        return l2_norm(x)


def se_resnet50_ir():
    model = IRSeResNet50(bottleneck_IR_SE, [3, 4, 14, 3])
    return model
