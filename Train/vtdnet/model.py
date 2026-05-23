from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from Train.vtdnet.blocks import CSPStage, ConvBNAct, DWConvBNAct


@dataclass(frozen=True)
class VTDNetConfig:
  """Конфигурация VTDNet (Vehicle Transport Detection Network)."""

  num_classes: int = 4
  imgsz: int = 640
  # Каналы backbone / FPN
  channels: tuple[int, int, int, int] = (48, 96, 192, 384)
  # Strides голов: 8, 16, 32 при imgsz=640 → сетки 80×80, 40×40, 20×20
  strides: tuple[int, int, int] = (8, 16, 32)


class VTDNet(nn.Module):
  """
  Собственная anchor-free архитектура (не YOLO).

  Backbone (DW-CSP) → FPN → три scale-aware головы.
  На каждую ячейку: objectness + bbox (cx,cy,w,h в [0,1]) + logits классов.
  """

  def __init__(self, cfg: VTDNetConfig | None = None) -> None:
    super().__init__()
    self.cfg = cfg or VTDNetConfig()
    c1, c2, c3, c4 = self.cfg.channels
    nc = self.cfg.num_classes
    self.out_channels = 5 + nc  # obj + 4 coords + classes

    self.stem = ConvBNAct(3, c1, kernel=3, stride=2)  # /2
    self.stage2 = CSPStage(c1, c1, depth=2, downsample=False)
    self.stage3 = CSPStage(c1, c2, depth=2, downsample=True)   # /4  → 160
    self.stage4 = CSPStage(c2, c3, depth=3, downsample=True)   # /8  → P3
    self.stage5 = CSPStage(c3, c4, depth=2, downsample=True)   # /16 → P4
    self.stage6 = CSPStage(c4, c4, depth=2, downsample=True)   # /32 → P5

    # FPN: P5 → P4 → P3 (выравнивание каналов перед сложением)
    self.lat5 = ConvBNAct(c4, c3, kernel=1)
    self.lat4 = ConvBNAct(c4, c3, kernel=1)
    self.lat3 = ConvBNAct(c3, c2, kernel=1)
    self.lat4_to_p3 = ConvBNAct(c3, c2, kernel=1)
    self.smooth4 = DWConvBNAct(c3)
    self.smooth3 = DWConvBNAct(c2)

    self.head3 = self._make_head(c2, self.out_channels)
    self.head4 = self._make_head(c3, self.out_channels)
    self.head5 = self._make_head(c4, self.out_channels)

    self._init_heads()

  def _make_head(self, in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
      DWConvBNAct(in_ch),
      ConvBNAct(in_ch, in_ch, kernel=1),
      nn.Conv2d(in_ch, out_ch, kernel_size=1),
    )

  def _init_heads(self) -> None:
    for head in (self.head3, self.head4, self.head5):
      conv = head[-1]
      assert isinstance(conv, nn.Conv2d)
      nn.init.constant_(conv.bias, -4.6)  # objectness ~0.01 после sigmoid

  def _forward_backbone(
    self, x: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = self.stem(x)
    x = self.stage2(x)
    x = self.stage3(x)
    p3 = self.stage4(x)   # stride 8
    p4 = self.stage5(p3)  # stride 16
    p5 = self.stage6(p4)  # stride 32
    return p3, p4, p5

  def _forward_fpn(
    self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p4 = self.smooth4(
      self.lat4(p4) + F.interpolate(self.lat5(p5), size=p4.shape[-2:], mode="nearest")
    )
    p3 = self.smooth3(
      self.lat3(p3)
      + F.interpolate(self.lat4_to_p3(p4), size=p3.shape[-2:], mode="nearest")
    )
    return p3, p4, p5

  def forward_heads(
    self, x: torch.Tensor
  ) -> list[torch.Tensor]:
    """Список тензоров (B, H*W, 5+nc) для каждого stride."""
    p3, p4, p5 = self._forward_backbone(x)
    p3, p4, p5 = self._forward_fpn(p3, p4, p5)

    outputs: list[torch.Tensor] = []
    for feat, head in ((p3, self.head3), (p4, self.head4), (p5, self.head5)):
      pred = head(feat)
      b, c, h, w = pred.shape
      outputs.append(pred.permute(0, 2, 3, 1).reshape(b, h * w, c))
    return outputs

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Обучение / инференс: (B, N, 5+nc), N = сумма ячеек всех уровней."""
    levels = self.forward_heads(x)
    return torch.cat(levels, dim=1)

  def forward_export(self, x: torch.Tensor) -> torch.Tensor:
    """Фиксированный выход для ONNX: сырые логиты, NMS на CPU/RKNN постобработке."""
    return self.forward(x)
