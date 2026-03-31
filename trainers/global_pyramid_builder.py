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


class GlobalPyramidBuilder(nn.Module):
    """
    Build multi-scale global features from selected ViT hidden states.

    Important fix compared with the current repo:
    config usually uses natural layer ids like [2, 5, 8, 11],
    while Python lists are zero-based. We therefore expose one_based_idx
    and default it to True.
    """

    def __init__(
        self,
        clip_dim=768,
        out_channels=(64, 128, 256, 512),
        selected_layers=(2, 5, 8, 11),
        one_based_idx=True,
    ):
        super().__init__()
        self.selected_layers = tuple(int(x) for x in selected_layers)
        self.one_based_idx = one_based_idx
        c1, c2, c3, c4 = out_channels

        self.lat1 = ConvGNAct(clip_dim, c1, kernel_size=1, stride=1, padding=0)
        self.lat2 = ConvGNAct(clip_dim, c2, kernel_size=1, stride=1, padding=0)
        self.lat3 = ConvGNAct(clip_dim, c3, kernel_size=1, stride=1, padding=0)
        self.lat4 = ConvGNAct(clip_dim, c4, kernel_size=1, stride=1, padding=0)

        self.td43 = ConvGNAct(c4, c3, kernel_size=1, stride=1, padding=0)
        self.td32 = ConvGNAct(c3, c2, kernel_size=1, stride=1, padding=0)
        self.td21 = ConvGNAct(c2, c1, kernel_size=1, stride=1, padding=0)

        self.smooth1 = ConvGNAct(c1, c1, kernel_size=3, stride=1, padding=1)
        self.smooth2 = ConvGNAct(c2, c2, kernel_size=3, stride=1, padding=1)
        self.smooth3 = ConvGNAct(c3, c3, kernel_size=3, stride=1, padding=1)
        self.smooth4 = ConvGNAct(c4, c4, kernel_size=3, stride=1, padding=1)

    def _tokens_to_map(self, hidden_state, B, H, W, patch_size):
        h_patch = H // patch_size
        w_patch = W // patch_size
        expected_tokens = h_patch * w_patch + 1

        if hidden_state.dim() != 3:
            raise ValueError(f"hidden_state must be 3D, got {hidden_state.shape}")

        if hidden_state.shape[0] == expected_tokens and hidden_state.shape[1] == B:
            hidden_state = hidden_state.permute(1, 0, 2).contiguous()
        elif hidden_state.shape[0] == B and hidden_state.shape[1] == expected_tokens:
            hidden_state = hidden_state.contiguous()
        else:
            raise ValueError(f"Unexpected hidden_state shape: {hidden_state.shape}")

        hidden_state = hidden_state[:, 1:, :]
        hidden_state = hidden_state.reshape(B, h_patch, w_patch, -1).permute(0, 3, 1, 2).contiguous()
        return hidden_state

    def _resolve_index(self, raw_idx: int, num_states: int) -> int:
        idx = raw_idx - 1 if self.one_based_idx else raw_idx
        return max(0, min(idx, num_states - 1))

    def forward(self, hidden_states, B, H, W, patch_size):
        if hidden_states is None:
            raise ValueError("hidden_states is required for GlobalPyramidBuilder")

        target_sizes = [
            (H // 4, W // 4),
            (H // 8, W // 8),
            (H // 16, W // 16),
            (H // 32, W // 32),
        ]

        num_states = len(hidden_states)
        maps = []
        for layer_idx in self.selected_layers:
            resolved = self._resolve_index(layer_idx, num_states)
            maps.append(self._tokens_to_map(hidden_states[resolved], B, H, W, patch_size))

        if len(maps) != 4:
            raise ValueError(f"Expected 4 selected feature maps, got {len(maps)}")

        m1 = F.interpolate(self.lat1(maps[0]), size=target_sizes[0], mode="bilinear", align_corners=False)
        m2 = F.interpolate(self.lat2(maps[1]), size=target_sizes[1], mode="bilinear", align_corners=False)
        m3 = F.interpolate(self.lat3(maps[2]), size=target_sizes[2], mode="bilinear", align_corners=False)
        m4 = F.interpolate(self.lat4(maps[3]), size=target_sizes[3], mode="bilinear", align_corners=False)

        g4 = self.smooth4(m4)
        g3 = self.smooth3(m3 + F.interpolate(self.td43(g4), size=m3.shape[-2:], mode="bilinear", align_corners=False))
        g2 = self.smooth2(m2 + F.interpolate(self.td32(g3), size=m2.shape[-2:], mode="bilinear", align_corners=False))
        g1 = self.smooth1(m1 + F.interpolate(self.td21(g2), size=m1.shape[-2:], mode="bilinear", align_corners=False))

        return [g1, g2, g3, g4]
