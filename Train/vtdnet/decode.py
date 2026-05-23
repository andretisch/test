from __future__ import annotations

import torch

from Train.constants import CLASS_NAMES, IMGSZ

# Strides FPN (должны совпадать с VTDNetConfig)
DEFAULT_STRIDES = (8, 16, 32)


def _build_cell_centers(
    imgsz: int,
    strides: tuple[int, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Нормированные (cx, cy) центры каждой ячейки, shape (N,)."""
    cx_list: list[torch.Tensor] = []
    cy_list: list[torch.Tensor] = []
    for stride in strides:
        gh = gw = imgsz // stride
        ys = torch.arange(gh, device=device, dtype=torch.float32)
        xs = torch.arange(gw, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        cx_list.append((((grid_x + 0.5) * stride) / imgsz).reshape(-1))
        cy_list.append((((grid_y + 0.5) * stride) / imgsz).reshape(-1))
    return torch.cat(cx_list), torch.cat(cy_list)


def decode_predictions(
    preds: torch.Tensor,
    conf_thres: float = 0.08,
    iou_thres: float = 0.45,
    max_det: int = 300,
    imgsz: int = IMGSZ,
    strides: tuple[int, ...] = DEFAULT_STRIDES,
    topk_candidates: int = 1000,
) -> list[torch.Tensor]:
    """
    preds: (B, N, 5+nc) — сырые логиты.
    Возвращает (M, 6): x1,y1,x2,y2,score,class в координатах 0..1.

  Anchor-free decode: score = objectness × class × centerness
  (центр bbox близок к центру ячейки — как FCOS).
    """
    device = preds.device
    cell_cx, cell_cy = _build_cell_centers(imgsz, strides, device)

    batch_results: list[torch.Tensor] = []
    for b in range(preds.shape[0]):
        p = preds[b]
        obj = p[:, 0].sigmoid()
        boxes = p[:, 1:5].sigmoid()
        cls_logits = p[:, 5:]
        cls_scores, cls_ids = cls_logits.softmax(dim=-1).max(dim=-1)

        bcx, bcy = boxes[:, 0], boxes[:, 1]
        # centerness: штраф, если предсказанный центр далеко от центра ячейки
        # centerness: подавляем срабатывания «чужих» ячеек (мягкий штраф)
        dist = ((bcx - cell_cx) ** 2 + (bcy - cell_cy) ** 2).sqrt() * imgsz
        centerness = torch.exp(-dist / 80.0)

        scores = obj * cls_scores * centerness

        k = min(topk_candidates, scores.numel())
        top_idx = scores.topk(k).indices
        mask = scores[top_idx] > conf_thres
        idx = top_idx[mask]

        if idx.numel() == 0:
            batch_results.append(torch.zeros((0, 6), device=device))
            continue

        b_boxes = boxes[idx]
        b_scores = scores[idx]
        b_cls = cls_ids[idx].float()

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
