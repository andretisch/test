from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox_ciou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
  """CIoU для bbox в формате cx,cy,w,h (норм. 0..1)."""
  px1 = pred[..., 0] - pred[..., 2] / 2
  py1 = pred[..., 1] - pred[..., 3] / 2
  px2 = pred[..., 0] + pred[..., 2] / 2
  py2 = pred[..., 1] + pred[..., 3] / 2

  tx1 = target[..., 0] - target[..., 2] / 2
  ty1 = target[..., 1] - target[..., 3] / 2
  tx2 = target[..., 0] + target[..., 2] / 2
  ty2 = target[..., 1] + target[..., 3] / 2

  inter_x1 = torch.max(px1, tx1)
  inter_y1 = torch.max(py1, ty1)
  inter_x2 = torch.min(px2, tx2)
  inter_y2 = torch.min(py2, ty2)

  inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
  area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
  area_t = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
  union = area_p + area_t - inter + 1e-7
  iou = inter / union

  cx_p = (px1 + px2) / 2
  cy_p = (py1 + py2) / 2
  cx_t = (tx1 + tx2) / 2
  cy_t = (ty1 + ty2) / 2

  cw = torch.max(px2, tx2) - torch.min(px1, tx1)
  ch = torch.max(py2, ty2) - torch.min(py1, ty1)
  c2 = cw**2 + ch**2 + 1e-7
  rho2 = (cx_p - cx_t) ** 2 + (cy_p - cy_t) ** 2

  return iou - rho2 / c2


class VTDNetLoss(nn.Module):
  def __init__(
    self,
    num_classes: int,
    obj_weight: float = 1.0,
    cls_weight: float = 1.0,
    box_weight: float = 2.0,
  ) -> None:
    super().__init__()
    self.num_classes = num_classes
    self.obj_weight = obj_weight
    self.cls_weight = cls_weight
    self.box_weight = box_weight
    self.bce_obj = nn.BCEWithLogitsLoss(reduction="none")
    self.ce_cls = nn.CrossEntropyLoss(reduction="none")

  def forward(
    self,
    preds: torch.Tensor,
    obj_t: torch.Tensor,
    cls_t: torch.Tensor,
    box_t: torch.Tensor,
    pos_mask: torch.Tensor,
  ) -> tuple[torch.Tensor, dict[str, float]]:
    obj_logits = preds[..., 0]
    box_pred = preds[..., 1:5].sigmoid()
    cls_logits = preds[..., 5:]

    loss_obj = self.bce_obj(obj_logits, obj_t).mean()

    n_pos = pos_mask.sum().clamp(min=1)
    loss_cls = self.ce_cls(
      cls_logits[pos_mask],
      cls_t[pos_mask],
    ).sum() / n_pos

    ciou = bbox_ciou(box_pred[pos_mask], box_t[pos_mask])
    loss_box = (1.0 - ciou).sum() / n_pos

    total = (
      self.obj_weight * loss_obj
      + self.cls_weight * loss_cls
      + self.box_weight * loss_box
    )
    stats = {
      "loss": float(total.detach()),
      "obj": float(loss_obj.detach()),
      "cls": float(loss_cls.detach()),
      "box": float(loss_box.detach()),
      "pos": int(pos_mask.sum().item()),
    }
    return total, stats
