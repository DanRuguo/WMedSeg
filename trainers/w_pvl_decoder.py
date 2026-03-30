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
    def __init__(self, channels, expand_ratio=2):
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


class ScopeBridge(nn.Module):
    """
    Keep local/global streams separate, but exchange information at each scale.
    """
    def __init__(self, channels, text_dim):
        super().__init__()
        self.local_gate = TextFiLM(text_dim, channels)
        self.global_gate = TextFiLM(text_dim, channels)

        self.local_mix = nn.Sequential(
            ConvGNAct(channels * 2, channels, kernel_size=1, stride=1, padding=0),
            ResidualDWBlock(channels),
        )

        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.attn_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            make_gn(channels),
            nn.Sigmoid(),
        )
        self.global_out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            make_gn(channels),
        )

        self.fuse = nn.Sequential(
            ConvGNAct(channels * 2, channels, kernel_size=1, stride=1, padding=0),
            ResidualDWBlock(channels),
        )

    def forward(self, local_feat, global_feat, text_feat):
        l = self.local_gate(local_feat, text_feat)
        g = self.global_gate(global_feat, text_feat)

        l_msg = self.local_mix(torch.cat([l, g], dim=1))
        local_out = l + l_msg

        q = self.q_proj(g)
        k = self.k_proj(l)
        v = self.v_proj(l)
        attn = self.attn_gate(q * k)
        global_out = g + self.global_out(attn * v)

        fused = self.fuse(torch.cat([local_out, global_out], dim=1))
        return local_out, global_out, fused


class UpFuseBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(in_ch + skip_ch, out_ch, kernel_size=1, stride=1, padding=0),
            ResidualDWBlock(out_ch),
            ResidualDWBlock(out_ch),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class WPVLDecoder(nn.Module):
    """
    W-shaped decoder:
    1st top-down decode
    2nd top-down refinement
    """
    def __init__(self, channels=(64, 128, 256, 512), text_dim=512, out_dim=512):
        super().__init__()
        c1, c2, c3, c4 = channels

        self.bridge1 = ScopeBridge(c1, text_dim)
        self.bridge2 = ScopeBridge(c2, text_dim)
        self.bridge3 = ScopeBridge(c3, text_dim)
        self.bridge4 = ScopeBridge(c4, text_dim)

        self.pre4 = nn.Sequential(
            ResidualDWBlock(c4),
            ResidualDWBlock(c4),
        )

        self.dec3_1 = UpFuseBlock(c4, c3, c3)
        self.dec2_1 = UpFuseBlock(c3, c2, c2)
        self.dec1_1 = UpFuseBlock(c2, c1, c1)

        self.ref4 = nn.Sequential(
            ConvGNAct(c4 * 2, c4, kernel_size=1, stride=1, padding=0),
            ResidualDWBlock(c4),
        )

        self.dec3_2 = UpFuseBlock(c4, c3 * 2, c3)
        self.dec2_2 = UpFuseBlock(c3, c2 * 2, c2)
        self.dec1_2 = UpFuseBlock(c2, c1 * 2, c1)

        self.out_proj = nn.Sequential(
            ConvGNAct(c1, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, out_dim, kernel_size=1, bias=False),
        )

    def forward(self, local_feats, global_feats, text_feat):
        c1, c2, c3, c4 = local_feats
        g1, g2, g3, g4 = global_feats

        _, _, f1 = self.bridge1(c1, g1, text_feat)
        _, _, f2 = self.bridge2(c2, g2, text_feat)
        _, _, f3 = self.bridge3(c3, g3, text_feat)
        _, _, f4 = self.bridge4(c4, g4, text_feat)

        # 1st decoder pass
        d4 = self.pre4(f4)
        d3 = self.dec3_1(d4, f3)
        d2 = self.dec2_1(d3, f2)
        d1 = self.dec1_1(d2, f1)

        # 2nd decoder pass (W-shaped refinement)
        r4 = self.ref4(torch.cat([d4, f4], dim=1))
        r3 = self.dec3_2(r4, torch.cat([f3, d3], dim=1))
        r2 = self.dec2_2(r3, torch.cat([f2, d2], dim=1))
        r1 = self.dec1_2(r2, torch.cat([f1, d1], dim=1))

        out = self.out_proj(r1)  # [B, 512, 56, 56]
        return out