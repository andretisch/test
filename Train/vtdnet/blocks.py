from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel, stride=stride, padding=padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConvBNAct(nn.Module):
    """Depthwise-separable блок — дружелюбен к NPU (меньше MAC, стандартные ops)."""

    def __init__(self, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = ConvBNAct(channels, channels, kernel=3, stride=stride, groups=channels)
        self.pw = ConvBNAct(channels, channels, kernel=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class CSPStage(nn.Module):
    """Лёгкий stage: несколько DW-блоков + проекция каналов."""

    def __init__(self, in_ch: int, out_ch: int, depth: int, downsample: bool = True) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        self.down = ConvBNAct(in_ch, out_ch, kernel=3, stride=stride) if downsample else None
        ch = out_ch if downsample else in_ch
        self.blocks = nn.Sequential(*[DWConvBNAct(ch) for _ in range(depth)])
        self.out_proj = (
            ConvBNAct(ch, out_ch, kernel=1) if not downsample and ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down is not None:
            x = self.down(x)
        x = self.blocks(x)
        return self.out_proj(x)
