"""
融合评分: CLIP美学分 + 内容丰富度 + 三分法构图 + 居中构图 + 深度层次惩罚 + 主体完整性惩罚 + 主体缺失惩罚

评分维度:
  1. aes_norm      (0.45): CLIP美学预测分, 归一化到[0,1]
  2. content_score (0.25): 内容丰富度(像素std + 边缘密度), 解决"纯天空/空白框"
  3. thirds_score  (0.15): 三分法构图 (降低权重，弱化边缘趋势)
  4. center_score  (0.15): 居中构图分 (提高权重，强化主体居中)
  惩罚:
  - depth_score        (-0.10): 深度层次惩罚 (负权重，抑制过深/过杂乱的背景纵深)
  - object_clip_penalty: 中小型物体被截断时扣分
  - missing_subject_penalty (-0.50): 新增：当全图有主体但当前候选框没框到主体时给予重罚
"""

import os
import sys
import argparse

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from filter_and_visualize import process_one
from aesthetic_test import load_models, aesthetic_rerank, resolve_img_ids

# ============================================================
# 【新增】引入你现有的 YOLO 检测器
# ============================================================
from ultralytics import YOLO  # 确保你有这个库

# 直接在这里初始化你的模型
YOLO_MODEL_PATH = "D:/AI_Models/yolov8n-seg.pt"
print(f"正在加载辅助 YOLO 模型: {YOLO_MODEL_PATH}")
_YOLO_DETECTOR = YOLO(YOLO_MODEL_PATH)

# ============================================================
# 配置 - 所有权重集中在这里方便调整
# ============================================================

OUTPUT_DIR = "data/output/final_ranker"

# 融合权重 (正向分)
W_AES     = 0.29   # CLIP美学分
W_CONTENT = 0.20   # 内容丰富度
W_THIRDS  = 0.22   # 三分法构图权重
W_CENTER  = 0.30   # 居中构图权重

# 惩罚权重
W_DEPTH_PENALTY   = 0.25  # 深度层次负向权重惩罚
W_CLIP_PENALTY    = 0.60  # 物体截断惩罚
W_MISSING_PENALTY = 0.50  # 新增：空框（全图有主体但本框未框到）的惩罚权重
W_YOLO_PENALTY    = 0.00  # 新增：YOLO严格惩罚权重 (针对人、背包、杯子等被截断的情况)

# 截断惩罚参数
LARGE_OBJECT_THRESHOLD   = 0.45  # 面积占比>=此值的实例视为大物体, 豁免惩罚
CLIP_PENALTY_WEIGHT      = 0.5   # 单个实例最大惩罚值
CLIP_FULL_PENALTY_AT     = 0.4   # 切掉40%时达到最大惩罚
CLIP_COVERAGE_TOLERANCE  = 0.97  # cov>=此值视为"基本完整", 不罚

# 内容丰富度参数
CONTENT_STD_SATURATE  = 60.0   # 像素std达到此值时content子项满分
CONTENT_EDGE_SATURATE = 15.0   # 边缘密度达到此值时content子项满分

# 三分法参数
THIRDS_POSITIONS = [1/3, 2/3]   # 三分线位置


# ============================================================
# 维度1: 内容丰富度
# ============================================================

def compute_content_score(crop_rgb):
    if crop_rgb is None or crop_rgb.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # (a) 像素标准差
    std_val = float(np.std(gray))
    std_score = min(1.0, std_val / CONTENT_STD_SATURATE)

    # (b) 边缘密度 (Laplacian绝对值均值)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    edge_density = float(np.mean(np.abs(lap)))
    edge_score = min(1.0, edge_density / CONTENT_EDGE_SATURATE)

    return 0.5 * std_score + 0.5 * edge_score


# ============================================================
# 辅助函数: 提取焦点坐标
# ============================================================


def _get_people_group_centroid_in_box(
        box,
        person_masks,
        img_h,
        img_w):

    if len(person_masks) == 0:
        return None

    x1 = max(0, int(box.x1))
    y1 = max(0, int(box.y1))
    x2 = min(img_w, int(box.x2))
    y2 = min(img_h, int(box.y2))

    if x2 <= x1 or y2 <= y1:
        return None

    total_mass = 0.0
    cx_sum = 0.0
    cy_sum = 0.0

    for mask in person_masks:

        crop = mask[y1:y2, x1:x2]

        area = crop.sum()

        if area < 20:
            continue

        ys, xs = np.where(crop > 0)

        if len(xs) == 0:
            continue

        cx = float(np.mean(xs))
        cy = float(np.mean(ys))

        cx_sum += cx * area
        cy_sum += cy * area

        total_mass += area

    if total_mass == 0:
        return None

    cx = cx_sum / total_mass
    cy = cy_sum / total_mass

    rx = cx / (x2 - x1)
    ry = cy / (y2 - y1)

    return rx, ry

def _get_subject_centroid_in_box(box, instance_masks, landscape_masks, img_h, img_w):
    x1 = max(0, int(box.x1))
    y1 = max(0, int(box.y1))
    x2 = min(img_w, int(box.x2))
    y2 = min(img_h, int(box.y2))

    if x2 <= x1 or y2 <= y1:
        return None

    box_h = y2 - y1
    box_w = x2 - x1

    best_inside = 0
    best_mask_crop = None

    for mask in instance_masks:
        if landscape_masks and _is_landscape_mask(mask, landscape_masks):
            continue
        total = mask.sum()
        if total == 0:
            continue
        total_ratio = total / (img_h * img_w)
        if total_ratio >= LARGE_OBJECT_THRESHOLD:
            continue

        crop = mask[y1:y2, x1:x2]
        inside = crop.sum()
        if inside > best_inside:
            best_inside = inside
            best_mask_crop = crop

    if best_mask_crop is not None and best_inside > 0:
        ys, xs = np.where(best_mask_crop > 0)
        cx_local = float(np.mean(xs))
        cy_local = float(np.mean(ys))
        rx = cx_local / box_w
        ry = cy_local / box_h
        return rx, ry

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
    rx = float(np.mean(xs)) / w
    ry = float(np.mean(ys)) / h
    return rx, ry


# ============================================================
# 维度2: 三分法构图
# ============================================================

def compute_thirds_score(box, instance_masks, landscape_masks, person_masks, img_rgb, img_h, img_w):
    x1 = max(0, int(box.x1))
    y1 = max(0, int(box.y1))
    x2 = min(img_w, int(box.x2))
    y2 = min(img_h, int(box.y2))
    
    if len(person_masks) > 0:
        centroid = _get_people_group_centroid_in_box(
            box, person_masks, img_h, img_w
        )
    else:
        centroid = _get_subject_centroid_in_box(
            box, instance_masks, landscape_masks, img_h, img_w
        )

    

    if centroid is not None:
        rx, ry = centroid
    else:
        crop = img_rgb[y1:y2, x1:x2]
        rx, ry = _edge_centroid_in_box(crop)

    thirds = THIRDS_POSITIONS
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


# ============================================================
# 维度3: 居中构图评分
# ============================================================

def compute_center_score(box, instance_masks, landscape_masks, person_masks, img_rgb, img_h, img_w):
    x1 = max(0, int(box.x1))
    y1 = max(0, int(box.y1))
    x2 = min(img_w, int(box.x2))
    y2 = min(img_h, int(box.y2))
    
    if len(person_masks) > 0:
        centroid = _get_people_group_centroid_in_box(
            box, person_masks, img_h, img_w
        )
    else:
        centroid = _get_subject_centroid_in_box(
            box, instance_masks, landscape_masks, img_h, img_w
        )

    

    if centroid is not None:
        rx, ry = centroid
    else:
        crop = img_rgb[y1:y2, x1:x2]
        rx, ry = _edge_centroid_in_box(crop)

    dist_to_center = ((rx - 0.5)**2 + (ry - 0.5)**2) ** 0.5
    score = float(np.exp(-8.0 * dist_to_center ** 2))
    return score


# ============================================================
# 维度4: 深度层次评分
# ============================================================

def compute_depth_score(box, depth_map, img_h, img_w):
    if depth_map is None:
        return 0.0

    x1 = max(0, int(box.x1))
    y1 = max(0, int(box.y1))
    x2 = min(img_w, int(box.x2))
    y2 = min(img_h, int(box.y2))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    crop_depth = depth_map[y1:y2, x1:x2]
    std_val = float(np.std(crop_depth))

    DEPTH_STD_SATURATE = 0.25
    return min(1.0, std_val / DEPTH_STD_SATURATE)


# ============================================================
# 主体完整性惩罚 / 新增：主体缺失检测
# ============================================================

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


def compute_object_clip_penalty(box, instance_masks, landscape_masks, img_area):
    if not instance_masks:
        return 0.0

    h, w = instance_masks[0].shape
    x1 = max(0, int(box.x1))
    y1 = max(0, int(box.y1))
    x2 = min(w, int(box.x2))
    y2 = min(h, int(box.y2))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    penalty = 0.0
    for mask in instance_masks:
        total = mask.sum()
        if total < 200:
            continue
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            mask_width = xs.max() - xs.min()
            # 如果物体长得极宽且靠下，直接判定为背景层，特赦！
            if mask_width > (w * 0.8) and ys.max() > (h * 0.7):
                continue
        if landscape_masks and _is_landscape_mask(mask, landscape_masks):
            continue
        if total / img_area >= LARGE_OBJECT_THRESHOLD:
            continue
        
        # -----------------------------------------------------------------
        # 🎨【新增留白盾：给饱满构图强制注入“呼吸感”】
        # -----------------------------------------------------------------
        # 拿到当前主体 Mask 的绝对物理边界（上下左右的最外层像素坐标）
        ys, xs = np.where(mask > 0)
        if len(xs) == 0: continue
        m_x1, m_x2 = xs.min(), xs.max()
        m_y1, m_y2 = ys.min(), ys.max()
        m_h = m_y2 - m_y1
        
        # 计算一个很小的呼吸边界（比如主体高度的 4%）
        padding_y = int(m_h * 0.04) 
        
        # 如果取景框的顶部（y1），距离杯子最顶端的距离小于这个 padding
        # 说明顶部太贴了、没有留白，触发微小惩罚，强制让位给更有呼吸感的构图
        edge_penalty = 0.0
        if abs(y1 - m_y1) < padding_y:
            # 距离越近，惩罚越重，最大惩罚 0.10（很温柔，刚好够用来微调排名）
            edge_penalty += 0.15 * (1.0 - (abs(y1 - m_y1) / padding_y))
            
        if abs(y2 - m_y2) < padding_y: # 同理，底部如果太贴也稍微约束一下
            edge_penalty += 0.03 * (1.0 - (abs(y2 - m_y2) / padding_y))
        # -----------------------------------------------------------------

        inter = mask[y1:y2, x1:x2].sum()
        if inter == 0:
            continue

        cov = inter / total
        if cov >= CLIP_COVERAGE_TOLERANCE:
            penalty = max(penalty, edge_penalty)
            continue

        clipped = 1.0 - cov
        this_pen = CLIP_PENALTY_WEIGHT * min(1.0, clipped / CLIP_FULL_PENALTY_AT)
        penalty = max(penalty, this_pen + edge_penalty)

    return penalty


def compute_people_clip_penalty(
        box,
        person_masks,
        img_h,
        img_w):

    if len(person_masks) == 0:
        return 0.0

    x1 = max(0, int(box.x1))
    y1 = max(0, int(box.y1))
    x2 = min(img_w, int(box.x2))
    y2 = min(img_h, int(box.y2))

    penalties = []

    for pmask in person_masks:

        total = pmask.sum()

        if total < 50:
            continue

        inside = pmask[y1:y2, x1:x2].sum()

        cov = inside / total

        penalties.append(cov)

    if len(penalties) == 0:
        return 0.0

    mean_cov = np.mean(penalties)

    return 1.0 - mean_cov


def compute_yolo_strict_penalty(box, img_rgb, instance_masks=None):
    """
    自适应宽容版 YOLO 安全锁：
    主要用于拦截“由于构图框太小，把人或杯子切了一大半”的恶性截断。
    对精致的特写边缘、微小切边进行赦免，配合前级深度融合 mask 使用。
    """
    results = _YOLO_DETECTOR(img_rgb, verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return 0.0

    img_h, img_w = img_rgb.shape[:2]
    x1, y1, x2, y2 = max(0, int(box.x1)), max(0, int(box.y1)), min(img_w, int(box.x2)), min(img_h, int(box.y2))
    
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
    scores = results.boxes.conf.cpu().numpy()
    
    max_penalty = 0.0
    STRICT_CLASS_IDS = {0, 24, 26, 39, 41, 63, 64, 65, 66, 67, 73}

    crop_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    crop_mask[y1:y2, x1:x2] = 1

    for yolo_box, cls, score in zip(boxes, cls_ids, scores):
        if score < 0.30:  # 稍微提高置信度，过滤不可靠的目标
            continue
            
        if cls in STRICT_CLASS_IDS:
            bx1, by1, bx2, by2 = yolo_box
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(img_w, bx2), min(img_h, by2)
            
            # 计算当前候选框与 YOLO 目标矩形框的交集面积
            ix1, iy1 = max(x1, bx1), max(y1, by1)
            ix2, iy2 = min(x2, bx2), min(y2, by2)
            inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            sb_area = (bx2 - bx1) * (by2 - by1)
            if sb_area == 0: continue
            
            inside_ratio = inter_area / sb_area

            # 如果这个取景框跟杯子/人根本没挨着，或者只是擦边（小于3%），不罚
            if inside_ratio <= 0.03:
                continue

            # --- 🚀 核心优化：如果匹配到高质量前级 Mask，使用松弛度的像素级判定 ---
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
                        
                        # 【放宽容忍度】：从 0.96 放宽到 0.88
                        # 工业取景允许高阶特写切掉一点点外围（比如贴纸、杯柄最外侧不显眼的边缘）
                        if pixel_inside_ratio >= 0.88:
                            current_penalty = 0.0
                        else:
                            # 只有真正切掉超过 12% 实体像素时，才进行梯度扣分
                            current_penalty = (1.0 - pixel_inside_ratio) * 1.0
                            
                        pixel_level_success = True
                        max_penalty = max(max_penalty, current_penalty)

            # --- 🚀 降级优化：智能矩形判定（放宽容度） ---
            if not pixel_level_success:
                # 【放宽容忍度】：从 0.93 放宽到 0.85
                # 只要杯子 85% 的矩形区域都在取景框里，就认为是个好特写，不予扣分！
                if inside_ratio >= 0.85:
                    current_penalty = 0.0
                elif 0.05 < inside_ratio < 0.85:
                    # 线性递增惩罚
                    current_penalty = (1.0 - inside_ratio) * 0.8
                else:
                    current_penalty = 0.0
                
                max_penalty = max(max_penalty, current_penalty)

            # --- 背包专属看门狗（保持你的原规则，但稍微平滑） ---
            if cls == 24 and inside_ratio > 0.5:
                max_penalty = max(max_penalty, 1.0)

    return float(max_penalty)
def check_missing_subject_penalty(
        box,
        person_masks,
        img_h,
        img_w):
    """
    原图存在人物/动物主体，
    当前框没有包含这些主体时，
    返回1.0（最终扣0.5分）
    """

    if len(person_masks) == 0:
        return 0.0

    x1 = max(0, int(box.x1))
    y1 = max(0, int(box.y1))
    x2 = min(img_w, int(box.x2))
    y2 = min(img_h, int(box.y2))

    if x2 <= x1 or y2 <= y1:
        return 1.0

    for pmask in person_masks:

        total = pmask.sum()

        if total == 0:
            continue

        inside = pmask[y1:y2, x1:x2].sum()

        coverage = inside / total

        # 至少保留30%
        if coverage >= 0.30:
            return 0.0

    return 1.0

# ============================================================
# 融合排序
# ============================================================

def fuse_and_rank(img_rgb, records, instance_masks, landscape_masks, person_masks, depth_map):
    print("person_masks =", len(person_masks))
    
    img_h, img_w = img_rgb.shape[:2]
    img_area = img_h * img_w
    
    # 🔍 【DEBUG 钩子：生成原始 Mask 预览图】
    # =========================================================================
    img_h, img_w = img_rgb.shape[:2]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 创建一张黑底画布，用来画融合前的所有原始主体
    orig_mask_vis = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    if instance_masks:
        for idx, m in enumerate(instance_masks):
            # 给每个独立的 mask 分配一个随机颜色（避开纯黑）
            color = [int(x) for x in np.random.randint(50, 255, size=3)]
            orig_mask_vis[m > 0] = color
            # 在 mask 的中心写上它的实例 ID
            ys, xs = np.where(m > 0)
            if len(xs) > 0:
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                cv2.putText(orig_mask_vis, f"ID:{idx}", (cx, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 保存原始状态
    cv2.imwrite(os.path.join(OUTPUT_DIR, "DEBUG_0_orig_masks.jpg"), cv2.cvtColor(orig_mask_vis, cv2.COLOR_RGB2BGR))

    # =========================================================================
    # 🔥【主体像素强力粘合盾】（保持你的物理融合逻辑）
    # =========================================================================
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
                
                dilated1 = cv2.dilate(base_m, np.ones((5, 5), np.uint8))
                dilated2 = cv2.dilate(compare_m, np.ones((5, 5), np.uint8))
                is_touching = np.logical_and(dilated1, compare_m).sum() > 0 or np.logical_and(dilated2, base_m).sum() > 0
                
                depth2 = np.median(depth_map[compare_m > 0]) if compare_m.sum() > 0 else 0
                same_depth = abs(depth1 - depth2) < 0.05
                
                if is_touching and same_depth:
                    base_m = np.logical_or(base_m, compare_m).astype(np.uint8)
                    used_indices.add(idx2)
                    print(f"➔ [物理凝聚] 成功将实例 {idx2} 并入实例 {idx1}")
            
            fused_masks.append(base_m)
            used_indices.add(idx1)
        
        instance_masks = fused_masks

    # =========================================================================
    # 🔍 【DEBUG 钩子：生成融合后的 Mask 预览图】
    # =========================================================================
    fused_mask_vis = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    for idx, m in enumerate(instance_masks):
        color = [int(x) for x in np.random.randint(50, 255, size=3)]
        fused_mask_vis[m > 0] = color
        ys, xs = np.where(m > 0)
        if len(xs) > 0:
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            cv2.putText(fused_mask_vis, f"Fused_ID:{idx}", (cx, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
    cv2.imwrite(os.path.join(OUTPUT_DIR, "DEBUG_1_fused_masks.jpg"), cv2.cvtColor(fused_mask_vis, cv2.COLOR_RGB2BGR))

    print("  [1/4] CLIP美学评分...")
    aesthetic_rerank(img_rgb, records)

    print("  [2/4] 内容丰富度...")
    for r in records:
        b = r["box"]
        x1, y1 = max(0, int(b.x1)), max(0, int(b.y1))
        x2, y2 = min(img_w, int(b.x2)), min(img_h, int(b.y2))
        crop = img_rgb[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None
        r["content_score"] = compute_content_score(crop)

    print("  [3/4] 三分法构图 + 居中构图评分...")
    for r in records:
        r["thirds_score"] = compute_thirds_score(
            r["box"], instance_masks, landscape_masks, person_masks, img_rgb, img_h, img_w
        )
        r["center_score"] = compute_center_score(
            r["box"], instance_masks, landscape_masks, person_masks, img_rgb, img_h, img_w
        )

    print("  [4/4] 深度层次提取 + 缺失惩罚 + 完整性惩罚...")
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
        # 计算新增的主体缺失惩罚标记 (0.0 或 1.0)
        r["missing_subject"] = check_missing_subject_penalty(
            r["box"], person_masks, img_h, img_w
        )
        
        r["yolo_strict_penalty"] = compute_yolo_strict_penalty(r["box"], img_rgb, instance_masks)
        
        has_person = len(person_masks) > 0
        if has_person:
            # 🧍 【人物类】强化中心点占比，降低三分法占比
            current_w_center = W_CENTER * 1.55   # 中心分权重放大 1.5 倍
            current_w_thirds = W_THIRDS * 0.5   # 三分法权重收敛
            current_w_aes = W_AES * 0.8  # 人物类适当降低美学分权重
            current_w_pen = W_CLIP_PENALTY
            current_w_depth = W_DEPTH_PENALTY * 2.0
        else:
            current_w_center = W_CENTER * 0.62   # 中心分权重收敛
            current_w_thirds = W_THIRDS * 1.34   # 三分法权重放大 1.6 倍
            current_w_aes = W_AES * 1.12  # 非人物类适当提升美学分权重
            current_w_pen = W_CLIP_PENALTY * 1.3  # 非人物类适当提升截断惩罚权重
            current_w_depth = W_DEPTH_PENALTY * 0.72  # 非人物类适当提升深度惩罚权重
            
            

    # aesthetic_score 归一化到 [0,1]
    aes_vals = np.array([r["aesthetic_score"] for r in records])
    aes_min, aes_max = aes_vals.min(), aes_vals.max()
    aes_range = aes_max - aes_min if aes_max > aes_min else 1e-6
    for r in records:
        r["aes_norm"] = (r["aesthetic_score"] - aes_min) / aes_range

    # 最终融合公式调整
    for r in records:
        r["final_score"] = (
            current_w_aes       * r["aes_norm"]
            + W_CONTENT * r["content_score"]
            + current_w_thirds  * r["thirds_score"]
            + current_w_center  * r["center_score"]
            - current_w_depth   * r["depth_score"]
            - current_w_pen     * r["object_clip_penalty"]
            - W_MISSING_PENALTY * r["missing_subject"]   # 新增：空框直接重扣 0.5 分
            - W_YOLO_PENALTY    * r["yolo_strict_penalty"]  # 新增：YOLO严格惩罚
        )

    return sorted(records, key=lambda r: r["final_score"], reverse=True)


# ============================================================
# 可视化
# ============================================================

# ============================================================
# 修改后的等比例缩放可视化 (防止画面被压缩)
# ============================================================

def draw_final_top_k(img_rgb, records, k=20, framing_img=None):
    crops = []
    
    # 设定每个小格子的标准画面内容尺寸 (不含上方黑色文字区)
    TARGET_W, TARGET_H = 160, 160
    
    for r in records[:k]:
        b = r["box"]
        x1, y1 = int(b.x1), int(b.y1)
        x2, y2 = int(b.x2), int(b.y2)
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
            
        # --- 核心改进：等比例缩放并居中补黑边 (Letterbox) ---
        h_crop, w_crop = crop.shape[:2]
        scale = min(TARGET_W / w_crop, TARGET_H / h_crop)
        new_w = int(w_crop * scale)
        new_h = int(h_crop * scale)
        
        # 先等比例缩放画面
        crop_resized = cv2.resize(crop, (new_w, new_h))
        
        # 创建标准的 160x160 黑色画布，将缩放后的画面居中贴上去
        canvas = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
        dx = (TARGET_W - new_w) // 2
        dy = (TARGET_H - new_h) // 2
        canvas[dy:dy+new_h, dx:dx+new_w] = crop_resized
        
        # 标签文本配置
        label1 = f"fin={r['final_score']:.2f} aes={r['aes_norm']:.2f}"
        label2 = f"con={r['content_score']:.2f} 3rd={r['thirds_score']:.2f}"
        
        msg_penalty = W_MISSING_PENALTY if r.get("missing_subject", 0.0) > 0 else 0.0
        label3 = f"ctr={r['center_score']:.2f} msg=-{msg_penalty:.1f} pen={r['object_clip_penalty']:.1f}"
        
        # 在上方扩充 48 像素的黑色区域用来写字
        canvas = cv2.copyMakeBorder(canvas, 48, 0, 0, 0,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(canvas, label1, (2, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
        cv2.putText(canvas, label2, (2, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
        cv2.putText(canvas, label3, (2, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
        crops.append(canvas)

    if framing_img is not None:
        # 对真值框（Ground Truth）同样做等比例缩放和黑边填充
        fh, fw = framing_img.shape[:2]
        f_scale = min(TARGET_W / fw, TARGET_H / fh)
        fn_w, fn_h = int(fw * f_scale), int(fh * f_scale)
        f_resized = cv2.resize(framing_img, (fn_w, fn_h))
        
        f_canvas = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
        fdx = (TARGET_W - fn_w) // 2
        fdy = (TARGET_H - fn_h) // 2
        f_canvas[fdy:fdy+fn_h, fdx:fdx+fn_w] = f_resized
        
        f_canvas = cv2.copyMakeBorder(f_canvas, 48, 0, 0, 0,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 255)) # 红色底边区分
        cv2.putText(f_canvas, "GT framing", (2, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        crops.insert(0, f_canvas)

    cols = 5
    rows = (len(crops) + cols - 1) // cols
    grid = np.zeros((rows * 208, cols * 160, 3), dtype=np.uint8)
    for i, c in enumerate(crops):
        r_idx, col_idx = divmod(i, cols)
        grid[r_idx * 208: r_idx * 208 + 208, col_idx * 160: col_idx * 160 + 160] = c
    return grid


# ============================================================
# 主流程 - 增加单独保存最佳取景框逻辑
# ============================================================

def process_one_final(img_id, args):
    print(f"\n===== [融合评分] 处理 {img_id} =====")

    # ============================================================
    # 【核心修复】使用 *tmp_args 接住所有可能多出来的返回值，防止 unpack 报错
    # ============================================================
    outputs = process_one(img_id, args)
    
    # 前 6 个是我们核心需要的变量
    img_rgb = outputs[0]
    final_records = outputs[1]
    framing_img = outputs[2]
    instance_masks = outputs[3]
    landscape_masks = outputs[4]
    person_masks = outputs[5]
    depth_map = outputs[6]

    img_h, img_w = img_rgb.shape[:2]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ============================================================
    # 🔍 【批量 DEBUG 1：生成该图片的「原始」Mask 预览】
    # ============================================================
    import copy
    # 备份一份原始 mask，防止被后面的融合逻辑覆盖
    orig_masks_backup = copy.deepcopy(instance_masks)
    
    orig_mask_vis = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    if orig_masks_backup:
        # 使用固定的随机种子，保证每张图的 ID 对应颜色相对稳定
        np.random.seed(42) 
        for idx, m in enumerate(orig_masks_backup):
            if m.sum() == 0: continue
            color = [int(x) for x in np.random.randint(50, 255, size=3)]
            orig_mask_vis[m > 0] = color
            ys, xs = np.where(m > 0)
            if len(xs) > 0:
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                cv2.putText(orig_mask_vis, f"ID:{idx}", (cx, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"{img_id}_debug_0_orig_masks.jpg"),
        cv2.cvtColor(orig_mask_vis, cv2.COLOR_RGB2BGR)
    )

    # ============================================================
    # 跑融合排序逻辑（内部会把破碎的杯子、柠檬片进行多模态强行粘合）
    # ============================================================
    ranked = fuse_and_rank(
        img_rgb, final_records, instance_masks, landscape_masks, person_masks, depth_map = depth_map
    )

    # ============================================================
    # 🔍 【批量 DEBUG 2：生成该图片「融合后」的 Mask 预览】
    # ============================================================
    fused_mask_vis = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    if instance_masks:
        np.random.seed(42) # 保持相同的颜色映射逻辑
        for idx, m in enumerate(instance_masks):
            if m.sum() == 0: continue
            color = [int(x) for x in np.random.randint(50, 255, size=3)]
            fused_mask_vis[m > 0] = color
            ys, xs = np.where(m > 0)
            if len(xs) > 0:
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                cv2.putText(fused_mask_vis, f"Fused:{idx}", (cx, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"{img_id}_debug_1_fused_masks.jpg"),
        cv2.cvtColor(fused_mask_vis, cv2.COLOR_RGB2BGR)
    )
    print(f"  [Debug产出] 原始与融合Mask已保存 -> {img_id}_debug_0/1.jpg")

    # ============================================================
    # 保持你原本的网格图输出
    # ============================================================
    grid_img = draw_final_top_k(img_rgb, ranked, k=20, framing_img=framing_img)
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"{img_id}_final_grid.jpg"),
        cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR)
    )

    # 单独切出第一名最佳取景框，不压缩、不拉伸、原图分辨率保存
    if len(ranked) > 0:
        best_box = ranked[0]["box"]
        
        bx1 = max(0, int(best_box.x1))
        by1 = max(0, int(best_box.y1))
        bx2 = min(img_w, int(best_box.x2))
        by2 = min(img_h, int(best_box.y2))
        
        best_crop = img_rgb[by1:by2, bx1:bx2]
        if best_crop.size > 0:
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, f"{img_id}_best_crop.jpg"),
                cv2.cvtColor(best_crop, cv2.COLOR_RGB2BGR)
            )
            print(f"   [新增产出] 独立最佳取景框（原比例超清）已保存 -> {OUTPUT_DIR}/{img_id}_best_crop.jpg")

    # 保持你原本的统计文本输出
    with open(os.path.join(OUTPUT_DIR, f"{img_id}_final_stats.txt"),
              "w", encoding="utf-8") as f:
        f.write(f"图片: {img_id}\n")
        f.write(f"候选框数量: {len(ranked)}\n")
        f.write(f"权重: W_AES={W_AES} W_CONTENT={W_CONTENT} "
                f"W_THIRDS={W_THIRDS} W_CENTER={W_CENTER} "
                f"W_DEP=-{W_DEPTH_PENALTY} W_CLIP_PEN={W_CLIP_PENALTY} W_MISSING_PEN=-{W_MISSING_PENALTY}\n\n")
        f.write("排名  final   aes    con    3rd    ctr    dep    pen    msg    "
                "rule   L  i    area   box\n")
        for idx, r in enumerate(ranked):
            b = r["box"]
            f.write(
                f"{idx+1:4d}  "
                f"{r['final_score']:.3f}  "
                f"{r['aes_norm']:.3f}  "
                f"{r['content_score']:.3f}  "
                f"{r['thirds_score']:.3f}  "
                f"{r['center_score']:.3f}  "
                f"{r['depth_score']:.3f}  "
                f"{r['object_clip_penalty']:.3f}  "
                f"{r['missing_subject']:.1f}  "
                f"{r['score']:.3f}  "
                f"{'Y' if r['is_landscape'] else 'N'}  "
                f"{r['subject_idx']:>3}  "
                f"{r['area_ratio']:.3f}  "
                f"({int(b.x1)},{int(b.y1)},{int(b.x2)},{int(b.y2)})\n"
            )

    print(f"完成! -> {OUTPUT_DIR}/{img_id}_final_grid.jpg")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default="A01",
                         help="图片ID, 支持单个/逗号分隔/all")
    parser.add_argument("--no_depth", action="store_true",
                         help="跳过depth模型(加速调试时使用)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    load_models()

    img_ids = resolve_img_ids(args.img)
    print(f"共 {len(img_ids)} 张图片: {img_ids}")

    for img_id in img_ids:
        try:
            process_one_final(img_id, args)
        except Exception as e:
            import traceback
            print(f"[ERROR] {img_id}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()