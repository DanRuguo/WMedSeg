import torch
import torch.nn as nn


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


class MixStyle(nn.Module):
    """
    Feature-statistics mixing for domain generalization.
    Applied only during training and only on early CNN features so the local
    branch does not overfit to source-domain scanner/style cues.
    """
    def __init__(self, p: float = 0.5, alpha: float = 0.3, eps: float = 1e-6):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.beta = torch.distributions.Beta(self.alpha, self.alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p <= 0 or x.size(0) <= 1:
            return x
        if torch.rand(1, device=x.device).item() >= self.p:
            return x

        B = x.size(0)
        mu = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        sigma = (var + self.eps).sqrt()
        x_norm = (x - mu) / sigma

        perm = torch.randperm(B, device=x.device)
        lmda = self.beta.sample((B, 1, 1, 1)).to(x.device, dtype=x.dtype)
        mu_mix = lmda * mu + (1.0 - lmda) * mu[perm]
        sigma_mix = lmda * sigma + (1.0 - lmda) * sigma[perm]
        return x_norm * sigma_mix + mu_mix


class LocalCNNEncoder(nn.Module):
    """
    Input:  [B, 3, 224, 224]
    Output:
        c1: [B,  64, 56, 56]
        c2: [B, 128, 28, 28]
        c3: [B, 256, 14, 14]
        c4: [B, 512,  7,  7]
    """
    def __init__(self, in_channels=3, channels=(64, 128, 256, 512), use_mixstyle: bool = False,
                 mixstyle_p: float = 0.5, mixstyle_alpha: float = 0.3):
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

        self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha) if use_mixstyle else nn.Identity()

    def forward(self, x):
        x = self.stem(x)
        x = self.mixstyle(x)

        c1 = self.stage1(self.stage1_down(x))
        c1 = self.mixstyle(c1)

        c2 = self.stage2(self.stage2_down(c1))
        c2 = self.mixstyle(c2)

        c3 = self.stage3(self.stage3_down(c2))
        c4 = self.stage4(self.stage4_down(c3))

        return [c1, c2, c3, c4]
