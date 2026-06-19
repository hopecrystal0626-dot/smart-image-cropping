

from __future__ import annotations
import random
from typing import Dict, List, Optional
from crop.bbox_utils import BBox


_DEFAULT_SCALE_GRID: dict[float, int] = {
    0.20: 12,
    0.25: 12,
    0.30: 12,
    0.35: 11,
    0.40: 11,
    0.50: 8,
}


def generate_candidates(
    img_w: int,
    img_h: int,
    jitter_ratio: float = 0.05,
    seed: int = 42,
    scale_grid: Optional[Dict[float, int]] = None,
) -> List[BBox]:
    rng = random.Random(seed)
    seen: set[tuple] = set()
    candidates: List[BBox] = []
    active_scale_grid = scale_grid or _DEFAULT_SCALE_GRID

    for scale, grid_n in active_scale_grid.items():
        crop_w = int(img_w * scale)
        crop_h = int(img_h * scale)

        if crop_w < 32 or crop_h < 32:
            continue
        
        if crop_w >= img_w or crop_h >= img_h:
            continue
        jitter_x = int(crop_w * jitter_ratio)
        jitter_y = int(crop_h * jitter_ratio)

        for i in range(grid_n):
            for j in range(grid_n):
                x = int(i * (img_w - crop_w) / (grid_n - 1)) if grid_n > 1 else 0
                y = int(j * (img_h - crop_h) / (grid_n - 1)) if grid_n > 1 else 0
                
                # jitter 扰动，避免纯规则网格漏掉边界附近最优位置
                if jitter_x > 0:
                    x += rng.randint(-jitter_x, jitter_x)
                if jitter_y > 0:
                    y += rng.randint(-jitter_y, jitter_y)

                x = max(0, min(x, img_w - crop_w))
                y = max(0, min(y, img_h - crop_h))

                key = (x, y, x + crop_w, y + crop_h)
                if key in seen:
                    continue
                seen.add(key)

                candidates.append(BBox(x, y, x + crop_w, y + crop_h, scale))

    return candidates