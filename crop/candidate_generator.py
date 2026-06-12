'''from crop.bbox_utils import BBox


def generate_candidates(img_w, img_h):

    scales = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.50,
    ]

    candidates = []

    for scale in scales:

        crop_w = int(img_w * scale)
        crop_h = int(img_h * scale)

        # ======================
        # 1. 自适应 stride（核心优化）
        # ======================
        stride_ratio = 0.12  # 推荐 10%~15%

        step_x = max(10, int(crop_w * stride_ratio))
        step_y = max(10, int(crop_h * stride_ratio))

        # ======================
        # 2. 大窗口稍微放宽 stride（避免太密浪费）
        # ======================
        if scale >= 0.5:
            step_x = max(step_x, 20)
            step_y = max(step_y, 20)

        # ======================
        # 3. sliding window
        # ======================
        for y in range(0, img_h - crop_h + 1, step_y):
            for x in range(0, img_w - crop_w + 1, step_x):

                candidates.append(
                    BBox(x, y, x + crop_w, y + crop_h, scale)
                )

        # ======================
        # 4. 额外 grid 保底（防止大窗口太少）
        # ======================
        if scale >= 0.5:

            grid_n = 4  # 4x4 保底采样

            for i in range(grid_n):
                for j in range(grid_n):

                    x = int(i * (img_w - crop_w) / max(1, grid_n - 1))
                    y = int(j * (img_h - crop_h) / max(1, grid_n - 1))

                    candidates.append(
                        BBox(x, y, x + crop_w, y + crop_h, scale)
                    )

    return candidates
'''

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