from __future__ import annotations

import torch

from Train.vtdnet.model import VTDNetConfig


def build_level_grid(
  batch_size: int,
  height: int,
  width: int,
  stride: int,
  device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Центры ячеек в нормированных координатах [0,1]."""
  ys = torch.arange(height, device=device, dtype=torch.float32)
  xs = torch.arange(width, device=device, dtype=torch.float32)
  grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
  cx = (grid_x + 0.5) * stride
  cy = (grid_y + 0.5) * stride
  return cx.reshape(-1), cy.reshape(-1)


def assign_targets(
  targets: list[torch.Tensor],
  cfg: VTDNetConfig,
  device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """
  targets[i]: (num_boxes, 5) = class, cx, cy, w, h (норм. 0..1).

  Возвращает для каждого уровня объединённые тензоры длиной N (все ячейки):
    obj_target (B,N), cls_target (B,N), box_target (B,N,4), pos_mask (B,N)
  """
  strides = cfg.strides
  imgsz = cfg.imgsz
  # Пороги размера объекта (в пикселях) для выбора уровня FPN
  size_limits = (imgsz / strides[0], imgsz / strides[1])

  level_shapes = [(imgsz // s, imgsz // s) for s in strides]
  n_cells = sum(h * w for h, w in level_shapes)

  batch_size = len(targets)
  obj_t = torch.zeros(batch_size, n_cells, device=device)
  cls_t = torch.zeros(batch_size, n_cells, dtype=torch.long, device=device)
  box_t = torch.zeros(batch_size, n_cells, 4, device=device)
  pos_mask = torch.zeros(batch_size, n_cells, dtype=torch.bool, device=device)

  offset = 0
  for level_idx, (stride, (gh, gw)) in enumerate(zip(strides, level_shapes)):
    n_level = gh * gw
    cell_cx, cell_cy = build_level_grid(batch_size, gh, gw, stride, device)
    cell_cx = cell_cx / imgsz
    cell_cy = cell_cy / imgsz

    for b in range(batch_size):
      gt = targets[b]
      if gt.numel() == 0:
        continue

      for box in gt:
        cls_id = int(box[0].item())
        bcx, bcy, bw, bh = box[1:5].tolist()
        box_px = max(bw, bh) * imgsz

        if level_idx == 0 and box_px >= size_limits[0]:
          continue
        if level_idx == 1 and (box_px < size_limits[0] or box_px >= size_limits[1]):
          continue
        if level_idx == 2 and box_px < size_limits[1]:
          continue

        # Ближайшая ячейка по центру объекта
        dist = (cell_cx - bcx) ** 2 + (cell_cy - bcy) ** 2
        cell_idx = int(dist.argmin().item()) + offset

        if pos_mask[b, cell_idx]:
          # Уже занята — оставляем более крупный объект (большая площадь)
          if bw * bh <= box_t[b, cell_idx, 2] * box_t[b, cell_idx, 3]:
            continue

        pos_mask[b, cell_idx] = True
        obj_t[b, cell_idx] = 1.0
        cls_t[b, cell_idx] = cls_id
        box_t[b, cell_idx] = torch.tensor([bcx, bcy, bw, bh], device=device)

    offset += n_level

  return obj_t, cls_t, box_t, pos_mask
