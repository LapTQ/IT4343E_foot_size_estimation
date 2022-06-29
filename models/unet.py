import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.sequential(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.MaxPool2d(2),
            Block(in_channels, out_channels)
        )

    def forward(self, x):
        return self.sequential(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_short = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3)
        self.conv = Block(in_channels, out_channels, in_channels // 2)

    def forward(self, x, x_short):
        x_short = self.conv_short(x_short)
        x = self.up(x)
        dy = x_short.size()[2] - x.size()[2]
        dx = x_short.size()[3] - x.size()[3]
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([x_short, x], dim=1)
        x = self.conv(x)
        return x


class Out(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):

    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.inp = Block(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.out = Out(64, n_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.inp(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        # x = self.softmax(x)

        return x


if __name__ == '__main__':
    net = UNet(3, 1)

    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

    net = deeplabv3_mobilenet_v3_large(pretrained=True)
    net.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)

    a = torch.ones((4, 3, 224, 224))
    print(net(a)['out'].shape)

