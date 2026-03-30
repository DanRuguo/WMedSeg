import torch
import torch.nn as nn
from torch.nn import functional as F


def make_gn(channels: int):
    for g in [8, 4, 2, 1]:
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False)
        self.norm = make_gn(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResidualDWBlock(nn.Module):
    def __init__(self, channels: int, expand_ratio: int = 2):
        super().__init__()
        hidden = channels * expand_ratio

        self.dw = ConvGNAct(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.pw1 = ConvGNAct(channels, hidden, kernel_size=1, stride=1, padding=0)
        self.pw2 = ConvGNAct(hidden, channels, kernel_size=1, stride=1, padding=0, act=False)
        self.out_act = nn.GELU()

    def forward(self, x):
        identity = x
        x = self.dw(x)
        x = self.pw1(x)
        x = self.pw2(x)
        x = x + identity
        x = self.out_act(x)
        return x


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            ResidualDWBlock(out_ch),
        )

    def forward(self, x):
        return self.block(x)


class LocalCNNEncoder(nn.Module):
    """
    Input:  [B, 3, 224, 224]
    Output:
        c1: [B,  64, 56, 56]
        c2: [B, 128, 28, 28]
        c3: [B, 256, 14, 14]
        c4: [B, 512,  7,  7]
    """
    def __init__(self, in_channels=3, channels=(64, 128, 256, 512)):
        super().__init__()
        c1, c2, c3, c4 = channels

        self.stem = nn.Sequential(
            ConvGNAct(in_channels, 32, kernel_size=3, stride=2, padding=1),   # 224 -> 112
            ResidualDWBlock(32),
        )

        self.stage1_down = DownsampleBlock(32, c1)   # 112 -> 56
        self.stage1 = nn.Sequential(
            ResidualDWBlock(c1),
            ResidualDWBlock(c1),
        )

        self.stage2_down = DownsampleBlock(c1, c2)   # 56 -> 28
        self.stage2 = nn.Sequential(
            ResidualDWBlock(c2),
            ResidualDWBlock(c2),
        )

        self.stage3_down = DownsampleBlock(c2, c3)   # 28 -> 14
        self.stage3 = nn.Sequential(
            ResidualDWBlock(c3),
            ResidualDWBlock(c3),
        )

        self.stage4_down = DownsampleBlock(c3, c4)   # 14 -> 7
        self.stage4 = nn.Sequential(
            ResidualDWBlock(c4),
            ResidualDWBlock(c4),
        )

    def forward(self, x):
        x = self.stem(x)

        c1 = self.stage1(self.stage1_down(x))
        c2 = self.stage2(self.stage2_down(c1))
        c3 = self.stage3(self.stage3_down(c2))
        c4 = self.stage4(self.stage4_down(c3))

        return [c1, c2, c3, c4]