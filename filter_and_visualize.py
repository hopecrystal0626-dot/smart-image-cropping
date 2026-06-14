"""
候选框去重 + 初步筛选 + 可视化

用法:
    python filter_and_visualize.py --img A01

输出:
    data/output/filter_test/A01_overlay.jpg   -- 所有保留候选框叠加图(半透明)
    data/output/filter_test/A01_grid.jpg      -- 候选框九宫格小图拼接(便于逐个查看)
    data/output/filter_test/A01_stats.txt     -- 统计信息(覆盖率分布等)
"""

import os
import sys
import argparse

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from crop.bbox_utils import BBox, compute_iou, clip_bbox
from proposal.proposal_generator import generate_all_proposals
from saliency.u2net_detector import U2NetDetector
from composition.panoptic_detector import PanopticDetector

try:
    from transformers import pipeline
    HAS_DEPTH = True
except ImportError:
    HAS_DEPTH = False


# ============================================================
# 配置
# ============================================================

DATA_DIR = "data/testA"
OUTPUT_DIR = "data/output/filter_test"

NMS_IOU_THRESH = 0.85          # 粗去重阈值(先去掉几乎重复的框)
SUBJECT_COVERAGE_THRESH = 0.6  # 主体覆盖率硬过滤阈值(低于此值认为严重截断主体)
KEEP_TOP_N = 100                # 初筛后保留数量

DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"


# ============================================================
# 工具函数
# ============================================================

def load_image(img_id):
    """读取原图, 返回 RGB"""
    path = os.path.join(DATA_DIR, f"{img_id}.jpg")
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"找不到图片: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def load_framing(img_id):
    """读取标准答案图(可能不存在)"""
    path = os.path.join(DATA_DIR, f"{img_id}_framing.jpg")
    if not os.path.exists(path):
        return None
    img_bgr = cv2.imread(path)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def get_saliency_mask(img_rgb):
    """
    u2net 显著性 mask
    假定返回 0~255 单通道, 尺寸与原图相同(否则会 resize)
    """
    detector = U2NetDetector()
    mask = detector.predict(img_rgb)

    if mask.shape[:2] != img_rgb.shape[:2]:
        mask = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]),
                           interpolation=cv2.INTER_LINEAR)

    # 归一化到 0~1
    mask_norm = mask.astype(np.float32)
    if mask_norm.max() > 1.5:
        mask_norm /= 255.0

    return mask_norm  # float32, HxW, 0~1


def get_panoptic_result(img_rgb):
    detector = PanopticDetector()
    result = detector.predict(img_rgb)
    segments = detector.get_segment_bboxes(result)
    seg_map = result["segmentation"].cpu().numpy()
    return result, segments, seg_map


def get_depth_map(img_rgb):
    """返回归一化深度图 (0~1, 值越大越近 -- 取决于模型约定, 此处不强行约定方向)"""
    if not HAS_DEPTH:
        return None

    from PIL import Image
    pipe = pipeline(task="depth-estimation", model=DEPTH_MODEL_NAME)
    image = Image.fromarray(img_rgb)
    result = pipe(image)
    depth = np.array(result["depth"]).astype(np.float32)

    if depth.shape[:2] != img_rgb.shape[:2]:
        depth = cv2.resize(depth, (img_rgb.shape[1], img_rgb.shape[0]),
                            interpolation=cv2.INTER_LINEAR)

    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)
    return depth


# ============================================================
# 主体 mask 融合
# ============================================================

# "人物/动物"类别: 这类主体一旦被框碰到却没完整包住(被砍头/断脚),
# 视觉上是严重缺陷, 需要独立、强力地惩罚, 不依赖 subject_idx 的选择
PERSON_LIKE_LABELS = {"person", "dog", "cat", "horse", "bird", "cow",
                       "sheep", "bear", "elephant", "zebra", "giraffe"}


def build_instance_masks(img_rgb, saliency_mask, seg_map, segments,
                         depth_map=None,
                         saliency_thresh=0.5,
                         depth_near_ratio=0.35,
                         min_instance_area_ratio=0.005):
    """
    返回:
        instance_masks  : List[np.ndarray]  核心主体候选实例
        landscape_masks : List[np.ndarray]  风景/环境元素 (含天空)
        sky_masks       : List[np.ndarray]  纯天空 (用于空洞惩罚)
        person_masks    : List[np.ndarray]  人物/动物等"不可截断"主体,
                                             用于独立的截断惩罚检测
    """
    h, w = img_rgb.shape[:2]
    img_area = h * w

    instance_masks = []   # 核心主体（如小车、小船、垃圾桶、显著性物体）
    landscape_masks = []  # 风景元素（如树木、草地、河流、湖面、道路、天空等）
    sky_masks = []        # 专门存放天空掩码，用于空洞检测
    person_masks = []     # 人物/动物, 用于独立截断惩罚

    # 严格的虚无背景黑名单：只过滤掉绝对没有构图美感贡献的纯色块
    BACKGROUND_LABELS = {
        "wall", "ceiling", "ceiling-merged", "wall-other-merged",
        "floor", "floor-wood"
    }

    def get_inst_mask(seg):
        seg_id = seg.get("id", None)
        if seg_id is not None:
            return (seg_map == seg_id).astype(np.uint8)
        x1, y1, x2, y2 = seg["bbox"]
        m = np.zeros((h, w), dtype=np.uint8)
        m[y1:y2, x1:x2] = 1
        return m

    # 解析 mask2former
    for seg in segments:
        label_lower = seg["label"].lower()
        if label_lower in BACKGROUND_LABELS:
            continue
        if seg.get("area", 0) < img_area * min_instance_area_ratio:
            continue

        mask = get_inst_mask(seg)

        # 0. 人物/动物: 独立收集, 同时也作为核心主体参与 subject_coverage
        if label_lower in PERSON_LIKE_LABELS:
            person_masks.append(mask)
            instance_masks.append(mask)
            continue

        # 1. 精准识别天空
        is_sky = "sky" in label_lower
        # 2. 识别其他具有美学构图贡献的自然/结构风景（采用模糊匹配，适配各类衍生标签）
        is_landscape = any(kw in label_lower for kw in [
            "tree", "grass", "river", "water", "mountain", "road",
            "pavement", "plant", "sea", "lake", "wood", "building", "hill"
        ])

        if is_sky:
            sky_masks.append(mask)
            landscape_masks.append(mask)  # 天空也属于广义风景
        elif is_landscape:
            landscape_masks.append(mask)
        else:
            instance_masks.append(mask)  # 剩下的归为普通核心主体

    # ---------- u2net: 显著性连通区域（坚决划归为核心主体） ----------
    sal_binary = (saliency_mask > saliency_thresh).astype(np.uint8)
    if sal_binary.sum() >= img_area * min_instance_area_ratio:
        num_labels, labels_im = cv2.connectedComponents(sal_binary)
        for label_id in range(1, num_labels):
            comp_mask = (labels_im == label_id).astype(np.uint8)
            if comp_mask.sum() >= img_area * min_instance_area_ratio:
                instance_masks.append(comp_mask)

    # ---------- depth: 最近的连通区域（坚决划归为核心主体） ----------
    if depth_map is not None:
        depth_near = (depth_map > (1 - depth_near_ratio)).astype(np.uint8)
        if depth_near.sum() >= img_area * min_instance_area_ratio:
            num_labels, labels_im = cv2.connectedComponents(depth_near)
            for label_id in range(1, num_labels):
                comp_mask = (labels_im == label_id).astype(np.uint8)
                if comp_mask.sum() >= img_area * min_instance_area_ratio:
                    instance_masks.append(comp_mask)

    if len(instance_masks) == 0:
        instance_masks.append(np.ones((h, w), dtype=np.uint8))

    return instance_masks, landscape_masks, sky_masks, person_masks


# ============================================================
# 去重 + 初筛
# ============================================================

def subject_coverage(box: BBox, instance_masks):
    """
    对每个实例 mask 分别计算 "框内该实例像素 / 该实例总像素",
    取最大值作为该框的主体覆盖率。

    返回:
        (best_coverage, best_idx)
    """
    h, w = instance_masks[0].shape

    x1 = max(0, int(box.x1))
    y1 = max(0, int(box.y1))
    x2 = min(w, int(box.x2))
    y2 = min(h, int(box.y2))

    if x2 <= x1 or y2 <= y1:
        return 0.0, -1

    best_cov = 0.0
    best_idx = -1

    for idx, mask in enumerate(instance_masks):
        total = mask.sum()
        if total == 0:
            continue
        inside = mask[y1:y2, x1:x2].sum()
        cov = inside / total
        if cov > best_cov:
            best_cov = cov
            best_idx = idx

    if best_idx == -1:
        # 没有命中任何实例(框内不包含 instance_masks 里任何一个实例的像素)。
        # 之前返回 1.0(视为"不惩罚"/满分), 会导致这类框的 composition_score
        # 系统性虚高(因为 cov=1.0 封顶), 即使它什么"主体"都没框住也排到最前面。
        # 改为 0.0: 这类框默认 cov=0, 后续 is_landscape_intent 判断里
        # land_cov > cov*0.8 = 0 几乎总能成立, 只要 land_cov>=0.25 就会
        # 被正确划入风景赛道(用风景维度评分), 而不是错误地停留在主体赛道
        # 却拿着虚高的 cov=1.0。
        return 0.0, -1

    return best_cov, best_idx


def nms_dedup(boxes, scores, iou_thresh=NMS_IOU_THRESH):
    """
    简单 NMS 去重: 按 score 从高到低, IoU > thresh 的框去掉(保留 score 高的)
    """
    order = np.argsort(scores)[::-1]
    keep = []
    suppressed = set()

    for idx in order:
        if idx in suppressed:
            continue
        keep.append(idx)
        for j in order:
            if j == idx or j in suppressed:
                continue
            if compute_iou(boxes[idx], boxes[j]) > iou_thresh:
                suppressed.add(j)

    return keep


def initial_filter(boxes, instance_masks, landscape_masks, sky_masks, person_masks,
                   img_w, img_h,
                   coverage_thresh=SUBJECT_COVERAGE_THRESH,
                   keep_top_n=KEEP_TOP_N):
    """
    更新点 (本版):
    1. 【新增】人物/动物截断独立惩罚 (person_clip_penalty):
       不依赖 subject_idx, 遍历所有 person_masks, 只要框"碰到但没完整
       包住"某个人物/动物, 就重罚。即使这个人不是 subject_idx 选中的
       那个实例, 也会被惩罚到, 避免"砍头/断脚"的框混进结果。

    2. 【调整】is_landscape_intent 改为相对判断:
       不再只看 land_cov 的绝对值, 而是要求 land_cov 明显大于 cov,
       避免"垃圾桶+一点天空"这种主体特写框被误判为风景意图框,
       从而抢占风景赛道配额、挤掉真正的主体特写。

    3. 【调整】配额改为动态:
       根据全图 landscape_masks 的总面积占比, 动态决定风景赛道在
       100 张里的配额比例(20%~50%), 而不是固定 40%。风景元素占主导
       的图(如海天为主的远景), 风景配额自动提高; 风景元素较少、
       主体占主导的图(如垃圾桶占据画面中心), 风景配额自动降低,
       把更多名额留给主体特写。
    """
    img_area = img_w * img_h
    img_cx, img_cy = img_w / 2, img_h / 2

    records = []

    for box in boxes:
        b = clip_bbox(box, img_w, img_h)
        if b.width <= 0 or b.height <= 0:
            continue

        # 1. 计算核心主体覆盖率 (box内包含的主体像素 / 全图该主体总像素)
        cov, subj_idx = subject_coverage(b, instance_masks)

        # 2. 针对 subject_idx 对应实例的截断惩罚 (原有逻辑)
        clip_penalty = 0.0
        if subj_idx is not None and subj_idx >= 0 and subj_idx < len(instance_masks):
            target_mask = instance_masks[subj_idx]
            total_subject_pixels = target_mask.sum()

            if total_subject_pixels > 0:
                inter_pixels = target_mask[int(b.y1):int(b.y2), int(b.x1):int(b.x2)].sum()
                clipped_ratio = 1.0 - cov

                if inter_pixels > 0 and clipped_ratio > 0.05:
                    clip_penalty = 0.6 * (clipped_ratio / 0.95)

        # 2b. 【新增】人物/动物独立截断惩罚
        # 不依赖 subj_idx, 遍历所有 person_masks, 任何一个被"碰到但未完整
        # 包住"都触发重罚。人物截断比一般物体更敏感(切掉>=30%即顶格)、
        # 惩罚力度更大(最高0.8分)。
        person_clip_penalty = 0.0
        for p_mask in person_masks:
            total = p_mask.sum()
            if total == 0:
                continue
            inter = p_mask[int(b.y1):int(b.y2), int(b.x1):int(b.x2)].sum()
            if inter == 0:
                continue  # 框完全没碰到这个人, 不惩罚
            p_cov = inter / total
            if p_cov < 0.98:
                clipped = 1.0 - p_cov
                this_penalty = 0.8 * min(1.0, clipped / 0.3)
                person_clip_penalty = max(person_clip_penalty, this_penalty)

        # 3. 计算风景环境总覆盖率
        land_cov = 0.0
        if len(landscape_masks) > 0:
            land_cov, _ = subject_coverage(b, landscape_masks)

        # 4. 计算纯天空占比
        sky_ratio_in_box = 0.0
        if len(sky_masks) > 0:
            sky_inter_pixels = sum([mask[int(b.y1):int(b.y2), int(b.x1):int(b.x2)].sum() for mask in sky_masks])
            sky_ratio_in_box = sky_inter_pixels / b.area if b.area > 0 else 0.0

        area_ratio = b.area / img_area

        # 原始面积惩罚得分逻辑
        if area_ratio < 0.12:
            area_score = area_ratio / 0.12
        elif area_ratio > 0.65:
            area_score = max(0.4, 1.0 - 0.4 * ((area_ratio - 0.65) / 0.35))
        else:
            area_score = 1.0

        # 中心偏移打分
        dx = abs(b.center_x - img_cx) / (img_w / 2)
        dy = abs(b.center_y - img_cy) / (img_h / 2)
        center_score = max(0.0, 1.0 - 0.4 * (dx + dy))

        # 【调整】区分赛道意图: 风景覆盖率不仅要够大(>=0.25), 还要明显
        # 超过主体覆盖率(land_cov > cov * 0.8), 否则仍视为主体特写框
        is_landscape_intent = (land_cov >= 0.25) and (land_cov > cov * 0.8)

        if is_landscape_intent:
            composition_score = 0.4 * cov + 0.6 * land_cov
        else:
            composition_score = cov

        # 综合粗筛分
        score = 0.5 * composition_score + 0.3 * area_score + 0.2 * center_score

        # 【扣分项 1】：纯天空空洞惩罚
        if sky_ratio_in_box >= 0.70:
            penalty = 0.5 * ((sky_ratio_in_box - 0.70) / 0.30)
            score = max(0.0, score - penalty)

        # 【扣分项 2】：主体/人体截断残缺惩罚 (原有, 针对 subject_idx)
        score = max(0.0, score - clip_penalty)

        # 【扣分项 3】：人物/动物独立截断惩罚 (新增, 不依赖 subject_idx)
        score = max(0.0, score - person_clip_penalty)

        records.append({
            "box": b,
            "coverage": cov,
            "land_coverage": land_cov,
            "subject_idx": subj_idx,
            "area_ratio": area_ratio,
            "is_landscape": is_landscape_intent,
            "clip_penalty": clip_penalty,
            "person_clip_penalty": person_clip_penalty,
            "score": score,
        })

    # 硬过滤: 主体截断 或 人物截断 严重的框, 在初筛阶段直接洗掉
    filtered = [
        r for r in records
        if r["person_clip_penalty"] < 0.2
        and ((r["coverage"] >= coverage_thresh and r["clip_penalty"] < 0.2) or
             (r["is_landscape"] and r["land_coverage"] >= 0.3 and r["clip_penalty"] < 0.2))
    ]
    if len(filtered) < keep_top_n:
        # 兜底放宽：要是全杀了导致不够100个, 才允许残缺框进来补位
        # (但仍优先按 score 排序, person_clip_penalty 已经计入 score 惩罚里)
        filtered = sorted(records, key=lambda x: x["score"], reverse=True)[:keep_top_n]

    # ---- 动态配额计算 ----
    # 风景元素在全图的总面积占比 (多个 landscape_masks 可能重叠, 用并集近似;
    # 这里用各 mask 求并集后的面积 / img_area, 作为"风景元素丰富度"的代理)
    if len(landscape_masks) > 0:
        landscape_union = np.zeros_like(landscape_masks[0])
        for m in landscape_masks:
            landscape_union = np.logical_or(landscape_union, m)
        total_landscape_ratio = landscape_union.sum() / img_area
    else:
        total_landscape_ratio = 0.0
    total_landscape_ratio = min(1.0, total_landscape_ratio)

    # 风景赛道配额比例: 基础20% + 风景占比贡献最多30%, 范围[0.2, 0.5]
    land_quota_ratio = 0.2 + 0.3 * total_landscape_ratio
    land_quota = int(round(keep_top_n * land_quota_ratio))
    sub_quota = keep_top_n - land_quota

    print(f"[DEBUG] 风景元素总面积占比={total_landscape_ratio:.3f} "
          f"-> 风景配额={land_quota}, 主体配额={sub_quota}")

    subject_channel = []
    landscape_channel = []

    for r in filtered:
        if r["is_landscape"]:
            landscape_channel.append(r)
        else:
            subject_channel.append(r)

    print(f"[DEBUG] 过滤后 -> 纯主体赛道: {len(subject_channel)} 个框 | 风景环境赛道: {len(landscape_channel)} 个框")

    # ---- 数量保底辅助函数 ----
    # 不引入任何新的几何/语义判断条件, 单纯用 filtered 中按 score 排序的
    # "次优框"补齐到 keep_top_n(用坐标判重, 避免重复)。
    def fill_to_quota(records_in, pool):
        if len(records_in) >= keep_top_n:
            return records_in[:keep_top_n] if len(records_in) > keep_top_n else records_in

        def box_key(r):
            b = r["box"]
            return (round(b.x1, 1), round(b.y1, 1), round(b.x2, 1), round(b.y2, 1))

        existing_keys = {box_key(r) for r in records_in}
        all_sorted = sorted(pool, key=lambda r: r["score"], reverse=True)

        out = list(records_in)
        for r in all_sorted:
            if len(out) >= keep_top_n:
                break
            k = box_key(r)
            if k not in existing_keys:
                out.append(r)
                existing_keys.add(k)

        out.sort(key=lambda r: r["score"], reverse=True)
        return out

    if len(landscape_channel) == 0 or len(subject_channel) == 0:
        print("[DEBUG] 警告：某一通道无合格候选框，触发老逻辑合并去重。")
        boxes_f = [r["box"] for r in filtered]
        scores_f = [r["score"] for r in filtered]
        keep_idx = nms_dedup(boxes_f, scores_f, NMS_IOU_THRESH)
        kept = [filtered[i] for i in keep_idx]
        kept.sort(key=lambda r: r["score"], reverse=True)
        kept = kept[:keep_top_n]
        kept = fill_to_quota(kept, records)
        print(f"[DEBUG] 最终输出框数量: {len(kept)} (目标 {keep_top_n})")
        return kept, records

    # ---- 独立赛道 A：纯主体聚焦框 NMS ----
    sub_boxes = [r["box"] for r in subject_channel]
    sub_scores = [r["score"] for r in subject_channel]
    sub_keep = nms_dedup(sub_boxes, sub_scores, NMS_IOU_THRESH)
    sub_kept = [subject_channel[i] for i in sub_keep]
    sub_kept.sort(key=lambda r: r["score"], reverse=True)

    # ---- 独立赛道 B：美学风景环境融合框 NMS ----
    land_boxes = [r["box"] for r in landscape_channel]
    land_scores = [r["score"] for r in landscape_channel]
    land_keep = nms_dedup(land_boxes, land_scores, iou_thresh=0.88)
    land_kept = [landscape_channel[i] for i in land_keep]
    land_kept.sort(key=lambda r: r["score"], reverse=True)

    # ---- 配额合并分配 (动态配额, 数量不足时互相补偿) ----
    final_land = land_kept[:land_quota]
    final_sub = sub_kept[:sub_quota]

    if len(final_land) < land_quota:
        extra = land_quota - len(final_land)
        final_sub = sub_kept[:sub_quota + extra]
    elif len(final_sub) < sub_quota:
        extra = sub_quota - len(final_sub)
        final_land = land_kept[:land_quota + extra]

    final_records = final_sub + final_land
    final_records.sort(key=lambda r: r["score"], reverse=True)

    final_records = fill_to_quota(final_records, records)

    print(f"[DEBUG] 最终输出框数量: {len(final_records)} (目标 {keep_top_n})")

    return final_records, records


# ============================================================
# 可视化
# ============================================================

def draw_overlay(img_rgb, records, framing_box=None):
    """
    所有保留框半透明叠加在原图上, 颜色按 score 渐变(红->绿)
    """
    img = img_rgb.copy()
    overlay = img.copy()

    scores = np.array([r["score"] for r in records])
    s_min, s_max = scores.min(), scores.max()
    norm = (scores - s_min) / (s_max - s_min + 1e-6)

    for r, n in zip(records, norm):
        b = r["box"]
        color = (
            int(255 * (1 - n)),  # R
            int(255 * n),         # G
            0
        )
        cv2.rectangle(overlay, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 1)

    blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

    if framing_box is not None:
        cv2.rectangle(blended,
                       (int(framing_box.x1), int(framing_box.y1)),
                       (int(framing_box.x2), int(framing_box.y2)),
                       (255, 255, 255), 3)

    return blended


def draw_top_k(img_rgb, records, k=12, framing_img=None):
    """
    把分数最高的 k 个框裁出来拼成网格图, 便于逐个查看效果
    """
    crops = []
    for r in records[:k]:
        if r.get("__separator__"):
            sep = np.full((180, 160, 3), (40, 80, 80), dtype=np.uint8)
            cv2.putText(sep, "SUBJECT", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(sep, "| LANDSCAPE", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            crops.append(sep)
            continue

        b = r["box"]
        crop = img_rgb[int(b.y1):int(b.y2), int(b.x1):int(b.x2)]
        crop = cv2.resize(crop, (160, 160))
        label = (f"s={r['score']:.2f} c={r['coverage']:.2f} "
                 f"i={r['subject_idx']} L={'Y' if r['is_landscape'] else 'N'}")
        crop = cv2.copyMakeBorder(crop, 20, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(crop, label, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
        crops.append(crop)

    if framing_img is not None:
        fcrop = cv2.resize(framing_img, (160, 160))
        fcrop = cv2.copyMakeBorder(fcrop, 20, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 255))
        cv2.putText(fcrop, "GT framing", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        crops.insert(0, fcrop)

    cols = 5
    rows = (len(crops) + cols - 1) // cols
    grid_h = rows * 180
    grid_w = cols * 160
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for i, c in enumerate(crops):
        r, col = divmod(i, cols)
        grid[r * 180: r * 180 + 180, col * 160: col * 160 + 160] = c

    return grid


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default="A01", help="图片 ID, 如 A01")
    parser.add_argument("--no_depth", action="store_true", help="跳过depth模型(加速调试)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_one(img_id, args):
    print(f"===== 处理 {img_id} =====")

    img_rgb = load_image(img_id)
    h, w = img_rgb.shape[:2]
    print(f"图片尺寸: {w}x{h}")

    framing_img = load_framing(img_id)

    # ---------- 模型推理 ----------
    print("u2net 显著性...")
    saliency_mask = get_saliency_mask(img_rgb)
    print(f"  saliency mask: min={saliency_mask.min():.3f} "
          f"max={saliency_mask.max():.3f} mean={saliency_mask.mean():.3f}")

    print("mask2former 语义分割...")
    pan_result, segments, seg_map = get_panoptic_result(img_rgb)
    print(f"  检测到 {len(segments)} 个实例")
    for s in segments[:5]:
        print(f"    {s['label']:15s} score={s.get('score', -1):.2f} area={s.get('area', -1)}")

    depth_map = None
    if not args.no_depth:
        print("depth anything v2 深度估计...")
        depth_map = get_depth_map(img_rgb)
        if depth_map is not None:
            print(f"  depth: min={depth_map.min():.3f} max={depth_map.max():.3f}")

    # ---------- 候选框生成 ----------
    print("生成候选框...")
    all_boxes = generate_all_proposals(img_rgb, saliency_mask * 255, segments)
    print(f"候选框总数: {len(all_boxes)}")

    # ---------- 主体实例 mask 列表 ----------
    print("提取候选主体实例...")
    instance_masks, landscape_masks, sky_masks, person_masks = build_instance_masks(
        img_rgb, saliency_mask, seg_map, segments, depth_map
    )
    print(f"  候选主体实例数: {len(instance_masks)}")
    for i, m in enumerate(instance_masks):
        print(f"    实例{i}: 面积占比={m.sum() / (h * w):.3f}")
    print(f"  风景元素数: {len(landscape_masks)}  (其中天空: {len(sky_masks)})")
    print(f"  人物/动物实例数: {len(person_masks)}")

    # ---------- 去重 + 初筛 ----------
    print("去重 + 初筛...")
    final_records, all_records = initial_filter(
        all_boxes, instance_masks, landscape_masks, sky_masks, person_masks, w, h
    )
    print(f"初筛后保留: {len(final_records)} 个框")

    coverages = np.array([r["coverage"] for r in all_records])
    print(f"覆盖率分布: min={coverages.min():.3f} "
          f"p25={np.percentile(coverages, 25):.3f} "
          f"median={np.median(coverages):.3f} "
          f"p75={np.percentile(coverages, 75):.3f} "
          f"max={coverages.max():.3f}")

    # ---------- 可视化 ----------
    print("生成可视化...")

    overlay_img = draw_overlay(img_rgb, final_records)
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"{img_id}_overlay.jpg"),
        cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
    )

    # 分别展示两个赛道各自的 Top10, 而不是合并排序后取 Top20。
    # 之前合并排序会出现"两个赛道分数区间接近"时, 一个赛道的高分框
    # 把另一个赛道的框全部挤出可视化范围的问题(即使配额机制本身生效,
    # 视觉上也看不到风景赛道里"其实有好框")。
    sub_records = [r for r in final_records if not r["is_landscape"]]
    land_records = [r for r in final_records if r["is_landscape"]]

    sub_records.sort(key=lambda r: r["score"], reverse=True)
    land_records.sort(key=lambda r: r["score"], reverse=True)

    print(f"[DEBUG] 最终100框中 -> 主体赛道: {len(sub_records)} | 风景赛道: {len(land_records)}")

    # 用一个纯色占位 crop 标记两个赛道的分界, 便于在网格图里区分
    separator = {"__separator__": True}

    display_records = sub_records[:10] + [separator] + land_records[:10]

    grid_img = draw_top_k(img_rgb, display_records, k=len(display_records), framing_img=framing_img)
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"{img_id}_topk_grid.jpg"),
        cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR)
    )

    # 主体实例 mask 可视化 (不同实例用不同颜色)
    instance_vis = np.zeros((h, w, 3), dtype=np.uint8)
    np.random.seed(0)
    inst_colors = np.random.randint(60, 255, (len(instance_masks), 3))
    for i, m in enumerate(instance_masks):
        instance_vis[m > 0] = inst_colors[i]

    mask_overlay = cv2.addWeighted(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 0.6,
        cv2.cvtColor(instance_vis, cv2.COLOR_RGB2BGR), 0.4, 0
    )
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"{img_id}_subject_mask.jpg"), mask_overlay
    )

    # 统计文件
    with open(os.path.join(OUTPUT_DIR, f"{img_id}_stats.txt"), "w", encoding="utf-8") as f:
        f.write(f"图片: {img_id}  尺寸: {w}x{h}\n")
        f.write(f"候选框总数(去重前): {len(all_boxes)}\n")
        f.write(f"初筛后保留: {len(final_records)}\n")
        f.write(f"候选主体实例数: {len(instance_masks)}\n")
        for i, m in enumerate(instance_masks):
            f.write(f"  实例{i}: 面积占比={m.sum() / (h * w):.4f}\n")
        f.write(f"风景元素数: {len(landscape_masks)} (天空: {len(sky_masks)})\n")
        f.write(f"人物/动物实例数: {len(person_masks)}\n")
        f.write(f"覆盖率分布: min={coverages.min():.3f} "
                f"p25={np.percentile(coverages, 25):.3f} "
                f"median={np.median(coverages):.3f} "
                f"p75={np.percentile(coverages, 75):.3f} "
                f"max={coverages.max():.3f}\n\n")
        f.write("Top 20 框详情:\n")
        for i, r in enumerate(final_records[:20]):
            b = r["box"]
            f.write(f"{i+1:2d}. box=({b.x1},{b.y1},{b.x2},{b.y2}) "
                    f"scale={b.scale} score={r['score']:.3f} "
                    f"coverage={r['coverage']:.3f} land_cov={r['land_coverage']:.3f} "
                    f"subject_idx={r['subject_idx']} is_landscape={r['is_landscape']} "
                    f"clip_pen={r['clip_penalty']:.2f} person_clip_pen={r['person_clip_penalty']:.2f} "
                    f"area_ratio={r['area_ratio']:.3f}\n")

    print(f"\n完成! 输出目录: {OUTPUT_DIR}")
    print(f"  - {img_id}_overlay.jpg      所有保留框叠加图")
    print(f"  - {img_id}_topk_grid.jpg    主体/风景两赛道Top10对比(含GT framing)")
    print(f"  - {img_id}_subject_mask.jpg 融合主体mask可视化")
    print(f"  - {img_id}_stats.txt        统计信息")


def resolve_img_ids(img_arg):
    """
    解析 --img 参数:
      - "A01"           -> ["A01"]
      - "A01,A02,A03"   -> ["A01", "A02", "A03"]
      - "all"           -> 扫描 DATA_DIR 下所有 A*.jpg (排除 *_framing.jpg)
    """
    if img_arg.lower() == "all":
        ids = []
        for fname in sorted(os.listdir(DATA_DIR)):
            if not fname.lower().endswith(".jpg"):
                continue
            if "_framing" in fname:
                continue
            ids.append(os.path.splitext(fname)[0])
        return ids

    return [s.strip() for s in img_arg.split(",") if s.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default="A01",
                         help="图片 ID, 支持单个(A01)、逗号分隔多个(A01,A02,A03)、"
                              "或 'all'(处理 data/testA 下所有 A*.jpg)")
    parser.add_argument("--no_depth", action="store_true", help="跳过depth模型(加速调试)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img_ids = resolve_img_ids(args.img)
    print(f"共 {len(img_ids)} 张图片待处理: {img_ids}\n")

    succeeded, failed = [], []

    for img_id in img_ids:
        try:
            process_one(img_id, args)
            succeeded.append(img_id)
        except Exception as e:
            print(f"[ERROR] 处理 {img_id} 失败: {e}")
            failed.append(img_id)
        print()  # 分隔每张图的日志

    print("===== 批量处理完成 =====")
    print(f"成功: {len(succeeded)} -> {succeeded}")
    if failed:
        print(f"失败: {len(failed)} -> {failed}")


if __name__ == "__main__":
    main()