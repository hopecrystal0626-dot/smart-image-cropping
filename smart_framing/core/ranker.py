# core/ranker.py
import cv2
import numpy as np
import os
from smart_framing.core.inference import aesthetic_score
from smart_framing import config

# ---------- YOLO 加载（可选） ----------
_YOLO_MODEL = None
def get_yolo():
    global _YOLO_MODEL
    if _YOLO_MODEL is None and os.path.exists(config.YOLO_MODEL_PATH):
        try:
            from ultralytics import YOLO
            _YOLO_MODEL = YOLO(config.YOLO_MODEL_PATH)
            print("YOLO loaded.")
        except:
            print("YOLO import failed, penalty disabled.")
            _YOLO_MODEL = False
    return _YOLO_MODEL

# ---------- 工具函数 ----------
def _is_landscape_mask(mask, landscape_masks, iou_thresh=0.3):
    total = mask.sum()
    if total == 0:
        return False
    for lm in landscape_masks:
        inter = np.logical_and(mask, lm).sum()
        union = np.logical_or(mask, lm).sum()
        if union > 0 and inter / union >= iou_thresh:
            return True
    return False

def _get_people_group_centroid_in_box(box, person_masks, img_h, img_w):
    if len(person_masks) == 0:
        return None
    x1 = max(0, int(box.x1)); y1 = max(0, int(box.y1))
    x2 = min(img_w, int(box.x2)); y2 = min(img_h, int(box.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    total_mass = 0.0; cx_sum = 0.0; cy_sum = 0.0
    for mask in person_masks:
        crop = mask[y1:y2, x1:x2]
        area = crop.sum()
        if area < 20: continue
        ys, xs = np.where(crop > 0)
        if len(xs) == 0: continue
        cx = float(np.mean(xs)); cy = float(np.mean(ys))
        cx_sum += cx * area; cy_sum += cy * area; total_mass += area
    if total_mass == 0: return None
    rx = (cx_sum / total_mass) / (x2 - x1)
    ry = (cy_sum / total_mass) / (y2 - y1)
    return rx, ry

def _get_subject_centroid_in_box(box, instance_masks, landscape_masks, img_h, img_w):
    x1 = max(0, int(box.x1)); y1 = max(0, int(box.y1))
    x2 = min(img_w, int(box.x2)); y2 = min(img_h, int(box.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    box_h = y2 - y1; box_w = x2 - x1
    best_inside = 0; best_mask_crop = None
    for mask in instance_masks:
        if landscape_masks and _is_landscape_mask(mask, landscape_masks):
            continue
        total = mask.sum()
        if total == 0: continue
        total_ratio = total / (img_h * img_w)
        if total_ratio >= config.LARGE_OBJECT_THRESHOLD:
            continue
        crop = mask[y1:y2, x1:x2]
        inside = crop.sum()
        if inside > best_inside:
            best_inside = inside
            best_mask_crop = crop
    if best_mask_crop is not None and best_inside > 0:
        ys, xs = np.where(best_mask_crop > 0)
        cx_local = float(np.mean(xs)); cy_local = float(np.mean(ys))
        return cx_local / box_w, cy_local / box_h
    return None

def _edge_centroid_in_box(crop_rgb):
    if crop_rgb is None or crop_rgb.size == 0:
        return 0.5, 0.5
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        return 0.5, 0.5
    h, w = crop_rgb.shape[:2]
    return float(np.mean(xs)) / w, float(np.mean(ys)) / h

# ---------- 各评分维度 ----------
def compute_content_score(crop_rgb):
    if crop_rgb is None or crop_rgb.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    std_val = float(np.std(gray))
    std_score = min(1.0, std_val / config.CONTENT_STD_SATURATE)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    edge_density = float(np.mean(np.abs(lap)))
    edge_score = min(1.0, edge_density / config.CONTENT_EDGE_SATURATE)
    return 0.5 * std_score + 0.5 * edge_score

def compute_thirds_score(box, instance_masks, landscape_masks, person_masks, img_rgb, img_h, img_w):
    x1 = max(0, int(box.x1)); y1 = max(0, int(box.y1))
    x2 = min(img_w, int(box.x2)); y2 = min(img_h, int(box.y2))
    if len(person_masks) > 0:
        centroid = _get_people_group_centroid_in_box(box, person_masks, img_h, img_w)
    else:
        centroid = _get_subject_centroid_in_box(box, instance_masks, landscape_masks, img_h, img_w)
    if centroid is not None:
        rx, ry = centroid
    else:
        crop = img_rgb[y1:y2, x1:x2]
        rx, ry = _edge_centroid_in_box(crop)
    thirds = config.THIRDS_POSITIONS
    min_dist = float('inf')
    for tx in thirds:
        for ty in thirds:
            dist = ((rx - tx)**2 + (ry - ty)**2) ** 0.5
            min_dist = min(min_dist, dist)
    dx = min(abs(rx - 1/3), abs(rx - 2/3))
    dy = min(abs(ry - 1/3), abs(ry - 2/3))
    line_dist = min(dx, dy)
    combined_dist = 0.5 * min_dist + 0.5 * line_dist
    score = float(np.exp(-15.0 * combined_dist ** 2))
    return score

def compute_center_score(box, instance_masks, landscape_masks, person_masks, img_rgb, img_h, img_w):
    x1 = max(0, int(box.x1)); y1 = max(0, int(box.y1))
    x2 = min(img_w, int(box.x2)); y2 = min(img_h, int(box.y2))
    if len(person_masks) > 0:
        centroid = _get_people_group_centroid_in_box(box, person_masks, img_h, img_w)
    else:
        centroid = _get_subject_centroid_in_box(box, instance_masks, landscape_masks, img_h, img_w)
    if centroid is not None:
        rx, ry = centroid
    else:
        crop = img_rgb[y1:y2, x1:x2]
        rx, ry = _edge_centroid_in_box(crop)
    dist_to_center = ((rx - 0.5)**2 + (ry - 0.5)**2) ** 0.5
    return float(np.exp(-8.0 * dist_to_center ** 2))

def compute_depth_score(box, depth_map, img_h, img_w):
    if depth_map is None:
        return 0.0
    x1 = max(0, int(box.x1)); y1 = max(0, int(box.y1))
    x2 = min(img_w, int(box.x2)); y2 = min(img_h, int(box.y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop_depth = depth_map[y1:y2, x1:x2]
    std_val = float(np.std(crop_depth))
    return min(1.0, std_val / config.DEPTH_STD_SATURATE)

def compute_object_clip_penalty(box, instance_masks, landscape_masks, img_area):
    if not instance_masks:
        return 0.0
    h, w = instance_masks[0].shape
    x1 = max(0, int(box.x1)); y1 = max(0, int(box.y1))
    x2 = min(w, int(box.x2)); y2 = min(h, int(box.y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    penalty = 0.0
    for mask in instance_masks:
        total = mask.sum()
        if total < 200: continue
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            mask_width = xs.max() - xs.min()
            if mask_width > (w * 0.8) and ys.max() > (h * 0.7):
                continue
        if landscape_masks and _is_landscape_mask(mask, landscape_masks):
            continue
        if total / img_area >= config.LARGE_OBJECT_THRESHOLD:
            continue
        # 留白盾
        ys, xs = np.where(mask > 0)
        if len(xs) == 0: continue
        m_x1, m_x2 = xs.min(), xs.max()
        m_y1, m_y2 = ys.min(), ys.max()
        m_h = m_y2 - m_y1
        padding_y = int(m_h * 0.04)
        edge_penalty = 0.0
        if abs(y1 - m_y1) < padding_y:
            edge_penalty += 0.15 * (1.0 - (abs(y1 - m_y1) / padding_y))
        if abs(y2 - m_y2) < padding_y:
            edge_penalty += 0.03 * (1.0 - (abs(y2 - m_y2) / padding_y))
        inter = mask[y1:y2, x1:x2].sum()
        if inter == 0:
            continue
        cov = inter / total
        if cov >= config.CLIP_COVERAGE_TOLERANCE:
            penalty = max(penalty, edge_penalty)
            continue
        clipped = 1.0 - cov
        this_pen = config.CLIP_PENALTY_WEIGHT * min(1.0, clipped / config.CLIP_FULL_PENALTY_AT)
        penalty = max(penalty, this_pen + edge_penalty)
    return penalty

def compute_people_clip_penalty(box, person_masks, img_h, img_w):
    if len(person_masks) == 0:
        return 0.0
    x1 = max(0, int(box.x1)); y1 = max(0, int(box.y1))
    x2 = min(img_w, int(box.x2)); y2 = min(img_h, int(box.y2))
    penalties = []
    for pmask in person_masks:
        total = pmask.sum()
        if total < 50: continue
        inside = pmask[y1:y2, x1:x2].sum()
        cov = inside / total
        penalties.append(cov)
    if len(penalties) == 0:
        return 0.0
    mean_cov = np.mean(penalties)
    return 1.0 - mean_cov

def compute_yolo_strict_penalty(box, img_rgb, instance_masks=None):
    model = get_yolo()
    if model is None or model is False:
        return 0.0
    try:
        results = model(img_rgb, verbose=False)[0]
    except:
        return 0.0
    if results.boxes is None or len(results.boxes) == 0:
        return 0.0
    img_h, img_w = img_rgb.shape[:2]
    x1 = max(0, int(box.x1)); y1 = max(0, int(box.y1))
    x2 = min(img_w, int(box.x2)); y2 = min(img_h, int(box.y2))
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
    scores = results.boxes.conf.cpu().numpy()
    max_penalty = 0.0
    STRICT_CLASS_IDS = {0, 24, 26, 39, 41, 63, 64, 65, 66, 67, 73}
    crop_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    crop_mask[y1:y2, x1:x2] = 1
    for yolo_box, cls, score in zip(boxes, cls_ids, scores):
        if score < 0.30:
            continue
        if cls in STRICT_CLASS_IDS:
            bx1, by1, bx2, by2 = yolo_box
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(img_w, bx2), min(img_h, by2)
            ix1, iy1 = max(x1, bx1), max(y1, by1)
            ix2, iy2 = min(x2, bx2), min(y2, by2)
            inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            sb_area = (bx2 - bx1) * (by2 - by1)
            if sb_area == 0: continue
            inside_ratio = inter_area / sb_area
            if inside_ratio <= 0.03:
                continue
            pixel_level_success = False
            if instance_masks is not None and len(instance_masks) > 0:
                best_mask = None
                max_overlap = 0
                for inst_m in instance_masks:
                    overlap_pts = np.sum(inst_m[by1:by2, bx1:bx2] > 0)
                    if overlap_pts > max_overlap:
                        max_overlap = overlap_pts
                        best_mask = inst_m
                if best_mask is not None and max_overlap > 40:
                    total_pixels = np.sum(best_mask > 0)
                    if total_pixels > 0:
                        inside_pixels = np.sum((best_mask > 0) & (crop_mask > 0))
                        pixel_inside_ratio = inside_pixels / total_pixels
                        if pixel_inside_ratio >= 0.88:
                            current_penalty = 0.0
                        else:
                            current_penalty = (1.0 - pixel_inside_ratio) * 1.0
                        pixel_level_success = True
                        max_penalty = max(max_penalty, current_penalty)
            if not pixel_level_success:
                if inside_ratio >= 0.85:
                    current_penalty = 0.0
                elif 0.05 < inside_ratio < 0.85:
                    current_penalty = (1.0 - inside_ratio) * 0.8
                else:
                    current_penalty = 0.0
                max_penalty = max(max_penalty, current_penalty)
            if cls == 24 and inside_ratio > 0.5:
                max_penalty = max(max_penalty, 1.0)
    return float(max_penalty)

def check_missing_subject_penalty(box, person_masks, img_h, img_w):
    if len(person_masks) == 0:
        return 0.0
    x1 = max(0, int(box.x1)); y1 = max(0, int(box.y1))
    x2 = min(img_w, int(box.x2)); y2 = min(img_h, int(box.y2))
    if x2 <= x1 or y2 <= y1:
        return 1.0
    for pmask in person_masks:
        total = pmask.sum()
        if total == 0: continue
        inside = pmask[y1:y2, x1:x2].sum()
        coverage = inside / total
        if coverage >= 0.30:
            return 0.0
    return 1.0

# ---------- 融合排序 ----------
def fuse_and_rank(img_rgb, records, instance_masks, landscape_masks, person_masks, depth_map):
    img_h, img_w = img_rgb.shape[:2]
    img_area = img_h * img_w

    # 主体像素粘合（如果深度可用）
    if instance_masks and len(instance_masks) > 1 and depth_map is not None:
        fused_masks = []
        used_indices = set()
        for idx1 in range(len(instance_masks)):
            if idx1 in used_indices: continue
            base_m = instance_masks[idx1].copy()
            depth1 = np.median(depth_map[base_m > 0]) if base_m.sum() > 0 else 0
            for idx2 in range(idx1 + 1, len(instance_masks)):
                if idx2 in used_indices: continue
                compare_m = instance_masks[idx2]
                dilated1 = cv2.dilate(base_m, np.ones((5,5), np.uint8))
                dilated2 = cv2.dilate(compare_m, np.ones((5,5), np.uint8))
                is_touching = (np.logical_and(dilated1, compare_m).sum() > 0) or (np.logical_and(dilated2, base_m).sum() > 0)
                depth2 = np.median(depth_map[compare_m > 0]) if compare_m.sum() > 0 else 0
                same_depth = abs(depth1 - depth2) < 0.05
                if is_touching and same_depth:
                    base_m = np.logical_or(base_m, compare_m).astype(np.uint8)
                    used_indices.add(idx2)
            fused_masks.append(base_m)
            used_indices.add(idx1)
        instance_masks = fused_masks

    # 计算各维度
    print("  [1/4] CLIP美学评分...")
    for r in records:
        b = r["box"]
        x1 = max(0, int(b.x1)); y1 = max(0, int(b.y1))
        x2 = min(img_w, int(b.x2)); y2 = min(img_h, int(b.y2))
        crop = img_rgb[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None
        r["aesthetic_score"] = aesthetic_score(crop) if crop is not None else 0.0

    print("  [2/4] 内容丰富度...")
    for r in records:
        b = r["box"]
        x1 = max(0, int(b.x1)); y1 = max(0, int(b.y1))
        x2 = min(img_w, int(b.x2)); y2 = min(img_h, int(b.y2))
        crop = img_rgb[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None
        r["content_score"] = compute_content_score(crop)

    print("  [3/4] 三分法 + 居中评分...")
    for r in records:
        r["thirds_score"] = compute_thirds_score(
            r["box"], instance_masks, landscape_masks, person_masks, img_rgb, img_h, img_w
        )
        r["center_score"] = compute_center_score(
            r["box"], instance_masks, landscape_masks, person_masks, img_rgb, img_h, img_w
        )

    print("  [4/4] 深度、缺失、完整性...")
    for r in records:
        r["depth_score"] = compute_depth_score(r["box"], depth_map, img_h, img_w)
        if len(person_masks) > 0:
            r["object_clip_penalty"] = compute_people_clip_penalty(
                r["box"], person_masks, img_h, img_w
            )
        else:
            r["object_clip_penalty"] = compute_object_clip_penalty(
                r["box"], instance_masks, landscape_masks, img_area
            )
        r["missing_subject"] = check_missing_subject_penalty(
            r["box"], person_masks, img_h, img_w
        )
        r["yolo_strict_penalty"] = compute_yolo_strict_penalty(r["box"], img_rgb, instance_masks)

    # 归一化 aesthetic_score
    aes_vals = np.array([r["aesthetic_score"] for r in records])
    aes_min, aes_max = aes_vals.min(), aes_vals.max()
    aes_range = aes_max - aes_min if aes_max > aes_min else 1e-6
    for r in records:
        r["aes_norm"] = (r["aesthetic_score"] - aes_min) / aes_range

    # 根据是否有人物调整权重
    has_person = len(person_masks) > 0
    if has_person:
        current_w_center = config.W_CENTER * 1.55
        current_w_thirds = config.W_THIRDS * 0.5
        current_w_aes = config.W_AES * 0.8
        current_w_depth = config.W_DEPTH_PENALTY * 2.0
        current_w_pen = config.W_CLIP_PENALTY
    else:
        current_w_center = config.W_CENTER * 0.62
        current_w_thirds = config.W_THIRDS * 1.34
        current_w_aes = config.W_AES * 1.12
        current_w_depth = config.W_DEPTH_PENALTY * 0.72
        current_w_pen = config.W_CLIP_PENALTY * 1.3

    for r in records:
        r["final_score"] = (
            current_w_aes * r["aes_norm"]
            + config.W_CONTENT * r["content_score"]
            + current_w_thirds * r["thirds_score"]
            + current_w_center * r["center_score"]
            - current_w_depth * r["depth_score"]
            - current_w_pen * r["object_clip_penalty"]
            - config.W_MISSING_PENALTY * r["missing_subject"]
            - config.W_YOLO_PENALTY * r["yolo_strict_penalty"]
        )

    return sorted(records, key=lambda r: r["final_score"], reverse=True)