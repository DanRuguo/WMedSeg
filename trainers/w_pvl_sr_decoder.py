import torch
import torch.nn as nn
from torch.nn import functional as F


def _num_groups(channels: int) -> int:
    for g in [16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class MixedNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gn = nn.GroupNorm(_num_groups(channels), channels)
        self.inorm = nn.InstanceNorm2d(channels, affine=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        a = torch.sigmoid(self.alpha)
        return a * self.gn(x) + (1.0 - a) * self.inorm(x)


class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False)
        self.norm = MixedNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResidualDWBlock(nn.Module):
    def __init__(self, channels, expand_ratio=2):
        super().__init__()
        hidden = channels * expand_ratio
        self.dw = ConvNormAct(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.pw1 = ConvNormAct(channels, hidden, kernel_size=1, stride=1, padding=0)
        self.pw2 = ConvNormAct(hidden, channels, kernel_size=1, stride=1, padding=0, act=False)
        self.out_act = nn.GELU()

    def forward(self, x):
        identity = x
        x = self.dw(x)
        x = self.pw1(x)
        x = self.pw2(x)
        return self.out_act(x + identity)


class TextFiLM(nn.Module):
    def __init__(self, text_dim, feat_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feat_dim * 2),
        )

    def forward(self, feat, text_feat):
        gamma, beta = self.mlp(text_feat).chunk(2, dim=-1)
        gamma = torch.tanh(gamma)
        return feat * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]


class ResidualLinearAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            MixedNorm2d(channels),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            MixedNorm2d(channels),
        )

    def forward(self, x, ref=None):
        ref = x if ref is None else ref
        attn = self.gate(self.q(x) * self.k(ref))
        return x + self.out(attn * self.v(ref))


class ScopeBridgeSR(nn.Module):
    def __init__(self, channels, text_dim):
        super().__init__()
        self.local_norm = MixedNorm2d(channels)
        self.global_norm = MixedNorm2d(channels)
        self.local_gate = TextFiLM(text_dim, channels)
        self.global_gate = TextFiLM(text_dim, channels)
        self.local_mix = nn.Sequential(
            ConvNormAct(channels * 2, channels, kernel_size=1, stride=1, padding=0),
            ResidualDWBlock(channels),
        )
        self.global_attn = ResidualLinearAttention(channels)
        self.fuse = nn.Sequential(
            ConvNormAct(channels * 2, channels, kernel_size=1, stride=1, padding=0),
            ResidualDWBlock(channels),
        )

    def forward(self, local_feat, global_feat, text_feat):
        l = self.local_gate(self.local_norm(local_feat), text_feat)
        g = self.global_gate(self.global_norm(global_feat), text_feat)
        l_out = l + self.local_mix(torch.cat([l, g], dim=1))
        g_out = self.global_attn(g, l)
        fused = self.fuse(torch.cat([l_out, g_out], dim=1))
        return fused


class DepthToSpaceUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, stride=1, padding=1, bias=False),
            MixedNorm2d(out_ch * 4),
            nn.GELU(),
        )
        self.shuffle = nn.PixelShuffle(2)
        self.refine = nn.Sequential(
            ConvNormAct(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            ResidualDWBlock(out_ch),
        )

    def forward(self, x):
        x = self.expand(x)
        x = self.shuffle(x)
        return self.refine(x)


class UpSRBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, text_dim, use_d2s=True):
        super().__init__()
        self.up = DepthToSpaceUp(in_ch, out_ch) if use_d2s else nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvNormAct(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )
        self.skip_gate = TextFiLM(text_dim, skip_ch)
        self.merge = nn.Sequential(
            ConvNormAct(out_ch + skip_ch, out_ch, kernel_size=1, stride=1, padding=0),
            ResidualDWBlock(out_ch),
        )
        self.attn = ResidualLinearAttention(out_ch)
        self.refine = nn.Sequential(
            ResidualDWBlock(out_ch),
            ResidualDWBlock(out_ch),
        )

    def forward(self, x, skip, text_feat):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.skip_gate(skip, text_feat)
        x = self.merge(torch.cat([x, skip], dim=1))
        x = self.attn(x, skip[:, : x.shape[1], ...] if skip.shape[1] == x.shape[1] else None)
        return self.refine(x)


class WPVLStyleRobustDecoder(nn.Module):
    """
    Style-robust W-PVL decoder.
    - Keeps the dual local/global fusion idea.
    - Uses mixed instance/group normalization to reduce scanner-style bias.
    - Uses depth-to-space in deep stages, inspired by MCADS.
    - Returns auxiliary projected feature maps for deep supervision.
    """

    def __init__(self, channels=(48, 96, 192, 384), text_dim=512, out_dim=512):
        super().__init__()
        c1, c2, c3, c4 = channels

        self.bridge1 = ScopeBridgeSR(c1, text_dim)
        self.bridge2 = ScopeBridgeSR(c2, text_dim)
        self.bridge3 = ScopeBridgeSR(c3, text_dim)
        self.bridge4 = ScopeBridgeSR(c4, text_dim)

        self.pre4 = nn.Sequential(
            ResidualDWBlock(c4),
            ResidualDWBlock(c4),
        )

        self.dec3_1 = UpSRBlock(c4, c3, c3, text_dim, use_d2s=True)
        self.dec2_1 = UpSRBlock(c3, c2, c2, text_dim, use_d2s=True)
        self.dec1_1 = UpSRBlock(c2, c1, c1, text_dim, use_d2s=False)

        self.ref4 = nn.Sequential(
            ConvNormAct(c4 * 2, c4, kernel_size=1, stride=1, padding=0),
            ResidualDWBlock(c4),
        )
        self.dec3_2 = UpSRBlock(c4, c3 * 2, c3, text_dim, use_d2s=False)
        self.dec2_2 = UpSRBlock(c3, c2 * 2, c2, text_dim, use_d2s=False)
        self.dec1_2 = UpSRBlock(c2, c1 * 2, c1, text_dim, use_d2s=False)

        self.out_proj = nn.Sequential(
            ConvNormAct(c1, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, out_dim, kernel_size=1, bias=False),
        )
        self.aux3 = nn.Conv2d(c3, out_dim, kernel_size=1, bias=False)
        self.aux2 = nn.Conv2d(c2, out_dim, kernel_size=1, bias=False)
        self.aux1 = nn.Conv2d(c1, out_dim, kernel_size=1, bias=False)

    def forward(self, local_feats, global_feats, text_feat):
        c1, c2, c3, c4 = local_feats
        g1, g2, g3, g4 = global_feats

        f1 = self.bridge1(c1, g1, text_feat)
        f2 = self.bridge2(c2, g2, text_feat)
        f3 = self.bridge3(c3, g3, text_feat)
        f4 = self.bridge4(c4, g4, text_feat)

        d4 = self.pre4(f4)
        d3 = self.dec3_1(d4, f3, text_feat)
        d2 = self.dec2_1(d3, f2, text_feat)
        d1 = self.dec1_1(d2, f1, text_feat)

        r4 = self.ref4(torch.cat([d4, f4], dim=1))
        r3 = self.dec3_2(r4, torch.cat([f3, d3], dim=1), text_feat)
        r2 = self.dec2_2(r3, torch.cat([f2, d2], dim=1), text_feat)
        r1 = self.dec1_2(r2, torch.cat([f1, d1], dim=1), text_feat)

        out = self.out_proj(r1)
        aux_feats = [self.aux3(r3), self.aux2(r2), self.aux1(r1)]
        return out, aux_feats
