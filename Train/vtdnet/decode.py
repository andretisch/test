from __future__ import annotations

import torch

from Train.constants import CLASS_NAMES


def decode_predictions(
  preds: torch.Tensor,
  conf_thres: float = 0.25,
  iou_thres: float = 0.45,
  max_det: int = 300,
) -> list[torch.Tensor]:
  """
  preds: (B, N, 5+nc) — сырые логиты с модели.
  Возвращает список (M, 6): x1,y1,x2,y2,score,class (пиксели 0..1).
  """
  obj = preds[..., 0].sigmoid()
  boxes = preds[..., 1:5].sigmoid()
  cls_logits = preds[..., 5:]
  cls_scores, cls_ids = cls_logits.softmax(dim=-1).max(dim=-1)
  scores = obj * cls_scores

  batch_results: list[torch.Tensor] = []
  for b in range(preds.shape[0]):
    mask = scores[b] > conf_thres
    if not mask.any():
      batch_results.append(torch.zeros((0, 6), device=preds.device))
      continue

    b_boxes = boxes[b][mask]
    b_scores = scores[b][mask]
    b_cls = cls_ids[b][mask].float()

    x1 = b_boxes[:, 0] - b_boxes[:, 2] / 2
    y1 = b_boxes[:, 1] - b_boxes[:, 3] / 2
    x2 = b_boxes[:, 0] + b_boxes[:, 2] / 2
    y2 = b_boxes[:, 1] + b_boxes[:, 3] / 2
    detections = torch.stack([x1, y1, x2, y2, b_scores, b_cls], dim=1)

    keep = nms_torch(detections[:, :4], detections[:, 4], iou_thres)
    detections = detections[keep]
    if len(detections) > max_det:
      detections = detections[:max_det]
    batch_results.append(detections)
  return batch_results


def nms_torch(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
  if boxes.numel() == 0:
    return torch.zeros(0, dtype=torch.long, device=boxes.device)

  order = scores.argsort(descending=True)
  keep: list[int] = []
  while order.numel() > 0:
    i = int(order[0])
    keep.append(i)
    if order.numel() == 1:
      break
    rest = order[1:]
    ious = box_iou(boxes[i].unsqueeze(0), boxes[rest]).squeeze(0)
    order = rest[ious <= iou_thres]

  return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
  inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
  inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
  inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
  inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])
  inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
  area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
  area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
  union = area1[:, None] + area2[None, :] - inter + 1e-7
  return inter / union
