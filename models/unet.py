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
        self.conv = Block(in_channels, out_channels, in_channels // 2)

    def forward(self, x, x_short):
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
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        return x


class UNet(nn.Module):

    def __init__(self, n_channels):
        super().__init__()
        self.inp = Block(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up11 = Up(1024, 256)
        self.up12 = Up(512, 128)
        self.up13 = Up(256, 64)
        self.up14 = Up(128, 64)
        self.out1 = Out(64, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.up21 = Up(1024, 256)
        self.up22 = Up(512, 128)
        self.up23 = Up(256, 64)
        self.up24 = Up(128, 64)
        self.out2 = Out(64, 1)
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inp(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x51 = self.up11(x5, x4)
        x51 = self.up12(x51, x3)
        x51 = self.up13(x51, x2)
        x51 = self.up14(x51, x1)
        x51 = self.out1(x51)
        x51 = self.sigmoid1(x51)
        x52 = self.up21(x5, x4)
        x52 = self.up22(x52, x3)
        x52 = self.up23(x52, x2)
        x52 = self.up24(x52, x1)
        x52 = self.out2(x52)
        x52 = self.sigmoid2(x52)

        return torch.cat([x51, x52], dim=1)


if __name__ == '__main__':
    unet = UNet(3)
    a = torch.ones((1, 3, 224, 224))
    print(unet(a).shape)
