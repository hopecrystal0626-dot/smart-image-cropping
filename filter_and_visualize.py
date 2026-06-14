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

def build_instance_masks(img_rgb, saliency_mask, seg_map, segments,
                          depth_map=None,
                          saliency_thresh=0.5,
                          depth_near_ratio=0.35,
                          min_instance_area_ratio=0.005):
    """
    生成"潜在主体实例 mask 列表", 而不是单一的"唯一主体 mask"。

    设计动机:
    一张图里可能同时存在多个"看起来都像主体"的物体(如远处的帆船 vs
    近处的龙虾笼), 模型(u2net/mask2former)的"显著性最高"判断不等于
    "摄影师真正想要的主体"。 与其武断地只选一个, 不如把所有合理的候选
    实例都纳入列表, 让候选框的"主体覆盖率"取对这些实例的最大值 ——
    这样无论真正的主体是哪一个, 覆盖它的候选框都不会被误过滤掉。

    实例来源:
    1. mask2former: 排除纯背景类(sky/wall等)后的所有实例
       (面积太小的噪声实例, < min_instance_area_ratio, 会被过滤掉)
    2. u2net: 显著性高于阈值的连通区域, 作为额外的"显著性主体"实例
       (即使它和mask2former的某个实例重叠也无妨, 重叠时取覆盖率较大者)
    3. depth (可选): 深度图中"最近"的连通区域, 作为"近景主体"实例

    返回:
        List[np.ndarray], 每个元素是 0/1 的 HxW mask
    """

    h, w = img_rgb.shape[:2]
    img_area = h * w

    instance_masks = []

    # ---------- mask2former: 所有非背景实例 ----------
    BACKGROUND_LABELS = {"sky-other-merged", "sky", "wall", "ceiling",
                          "floor", "floor-wood", "pavement-merged", "road",
                          "ceiling-merged", "wall-other-merged"}

    def get_inst_mask(seg):
        seg_id = seg.get("id", None)
        if seg_id is not None:
            return (seg_map == seg_id).astype(np.uint8)
        x1, y1, x2, y2 = seg["bbox"]
        m = np.zeros((h, w), dtype=np.uint8)
        m[y1:y2, x1:x2] = 1
        return m

    for seg in segments:
        if seg["label"].lower() in BACKGROUND_LABELS:
            continue
        if seg.get("area", 0) < img_area * min_instance_area_ratio:
            continue
        instance_masks.append(get_inst_mask(seg))

    # ---------- u2net: 显著性连通区域 ----------
    sal_binary = (saliency_mask > saliency_thresh).astype(np.uint8)
    if sal_binary.sum() >= img_area * min_instance_area_ratio:
        num_labels, labels_im = cv2.connectedComponents(sal_binary)
        for label_id in range(1, num_labels):
            comp_mask = (labels_im == label_id).astype(np.uint8)
            if comp_mask.sum() >= img_area * min_instance_area_ratio:
                instance_masks.append(comp_mask)

    # ---------- depth: 最近的连通区域 (可选) ----------
    if depth_map is not None:
        depth_near = (depth_map > (1 - depth_near_ratio)).astype(np.uint8)
        if depth_near.sum() >= img_area * min_instance_area_ratio:
            num_labels, labels_im = cv2.connectedComponents(depth_near)
            for label_id in range(1, num_labels):
                comp_mask = (labels_im == label_id).astype(np.uint8)
                if comp_mask.sum() >= img_area * min_instance_area_ratio:
                    instance_masks.append(comp_mask)

    if len(instance_masks) == 0:
        # 兜底: 整图作为唯一"主体", 此时覆盖率约束相当于失效
        instance_masks.append(np.ones((h, w), dtype=np.uint8))

    return instance_masks


# ============================================================
# 去重 + 初筛
# ============================================================

def subject_coverage(box: BBox, instance_masks):
    """
    对每个实例 mask 分别计算 "框内该实例像素 / 该实例总像素",
    取最大值作为该框的主体覆盖率。

    这样无论真正的"主体"是 instance_masks 里的哪一个,
    只要框覆盖了其中某一个完整的实例, 覆盖率就会高。

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
        return 1.0, -1  # 没有任何有效实例, 不惩罚

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


def initial_filter(boxes, instance_masks, img_w, img_h,
                    coverage_thresh=SUBJECT_COVERAGE_THRESH,
                    keep_top_n=KEEP_TOP_N):
    """
    1. clip 边界
    2. 计算主体覆盖率 (对多个候选实例取最大值)
    3. 计算一个简单的"构图分" (中心位置 + 面积比例), 用于打分排序
    4. 先按 coverage_thresh 硬过滤
    5. NMS 去重
    6. 按综合分排序, 取前 keep_top_n
    """

    img_area = img_w * img_h
    img_cx, img_cy = img_w / 2, img_h / 2

    records = []

    for box in boxes:
        b = clip_bbox(box, img_w, img_h)
        if b.width <= 0 or b.height <= 0:
            continue

        cov, subj_idx = subject_coverage(b, instance_masks)

        area_ratio = b.area / img_area
        # 面积比例打分: 偏好 0.15~0.5 之间, 太小或太大都扣分
        if area_ratio < 0.1:
            area_score = area_ratio / 0.1
        elif area_ratio > 0.7:
            area_score = max(0, 1 - (area_ratio - 0.7) / 0.3)
        else:
            area_score = 1.0

        # 中心偏移打分: 离图像中心越近, 分越高 (粗略, 实际可换成三分法等)
        dx = abs(b.center_x - img_cx) / (img_w / 2)
        dy = abs(b.center_y - img_cy) / (img_h / 2)
        center_score = 1.0 - 0.5 * (dx + dy)
        center_score = max(0.0, center_score)

        # 综合分 (此处仅用于初筛排序, 非最终评分)
        score = 0.6 * cov + 0.25 * area_score + 0.15 * center_score

        records.append({
            "box": b,
            "coverage": cov,
            "subject_idx": subj_idx,
            "area_ratio": area_ratio,
            "score": score,
        })

    # 硬过滤: 覆盖率过低的框 (严重截断主体)
    filtered = [r for r in records if r["coverage"] >= coverage_thresh]

    # 若硬过滤后数量太少 (比如全部图都被过滤), 放宽阈值
    if len(filtered) < keep_top_n:
        filtered = records

    boxes_f = [r["box"] for r in filtered]
    scores_f = [r["score"] for r in filtered]

    keep_idx = nms_dedup(boxes_f, scores_f, NMS_IOU_THRESH)

    kept = [filtered[i] for i in keep_idx]
    kept.sort(key=lambda r: r["score"], reverse=True)

    final = kept[:keep_top_n]

    return final, records


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
        b = r["box"]
        crop = img_rgb[int(b.y1):int(b.y2), int(b.x1):int(b.x2)]
        crop = cv2.resize(crop, (160, 160))
        label = f"s={r['score']:.2f} c={r['coverage']:.2f} i={r['subject_idx']}"
        crop = cv2.copyMakeBorder(crop, 20, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(crop, label, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
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

    img_id = args.img
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
    instance_masks = build_instance_masks(img_rgb, saliency_mask, seg_map, segments, depth_map)
    print(f"  候选主体实例数: {len(instance_masks)}")
    for i, m in enumerate(instance_masks):
        print(f"    实例{i}: 面积占比={m.sum() / (h * w):.3f}")

    # ---------- 去重 + 初筛 ----------
    print("去重 + 初筛...")
    final_records, all_records = initial_filter(all_boxes, instance_masks, w, h)
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

    grid_img = draw_top_k(img_rgb, final_records, k=20, framing_img=framing_img)
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
                    f"coverage={r['coverage']:.3f} subject_idx={r['subject_idx']} "
                    f"area_ratio={r['area_ratio']:.3f}\n")

    print(f"\n完成! 输出目录: {OUTPUT_DIR}")
    print(f"  - {img_id}_overlay.jpg      所有保留框叠加图")
    print(f"  - {img_id}_topk_grid.jpg    Top20框裁剪拼接(含GT framing对比)")
    print(f"  - {img_id}_subject_mask.jpg 融合主体mask可视化")
    print(f"  - {img_id}_stats.txt        统计信息")


if __name__ == "__main__":
    main()
    
"""
候选框去重 + 初步筛选 + 可视化

用法:
    python filter_and_visualize.py --img A01

输出:
    data/output/filter_test/A01_overlay.jpg   -- 所有保留候选框叠加图(半透明)
    data/output/filter_test/A01_grid.jpg      -- 候选框九宫格小图拼接(便于逐个查看)
    data/output/filter_test/A01_stats.txt     -- 统计信息(覆盖率分布等)
"""
'''
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
SUBJECT_COVERAGE_THRESH = 0.3   # 主体覆盖率硬过滤阈值(低于此值认为严重截断主体, 直接剔除)
MIN_AREA_RATIO = 0.02           # 候选框面积占比下限(过小的几乎不可用)
MAX_AREA_RATIO = 0.95           # 候选框面积占比上限(几乎是原图缩略图, 没有裁剪意义)
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

def build_instance_masks(img_rgb, saliency_mask, seg_map, segments,
                          depth_map=None,
                          saliency_thresh=0.5,
                          depth_near_ratio=0.35,
                          min_instance_area_ratio=0.005):
    """
    生成"潜在主体实例 mask 列表", 而不是单一的"唯一主体 mask"。

    设计动机:
    一张图里可能同时存在多个"看起来都像主体"的物体(如远处的帆船 vs
    近处的龙虾笼), 模型(u2net/mask2former)的"显著性最高"判断不等于
    "摄影师真正想要的主体"。 与其武断地只选一个, 不如把所有合理的候选
    实例都纳入列表, 让候选框的"主体覆盖率"取对这些实例的最大值 ——
    这样无论真正的主体是哪一个, 覆盖它的候选框都不会被误过滤掉。

    实例来源:
    1. mask2former: 排除纯背景类(sky/wall等)后的所有实例
       (面积太小的噪声实例, < min_instance_area_ratio, 会被过滤掉)
    2. u2net: 显著性高于阈值的连通区域, 作为额外的"显著性主体"实例
       (即使它和mask2former的某个实例重叠也无妨, 重叠时取覆盖率较大者)
    3. depth (可选): 深度图中"最近"的连通区域, 作为"近景主体"实例

    返回:
        List[np.ndarray], 每个元素是 0/1 的 HxW mask
    """

    h, w = img_rgb.shape[:2]
    img_area = h * w

    instance_masks = []

    # ---------- mask2former: 所有非背景实例 ----------
    BACKGROUND_LABELS = {"sky-other-merged", "sky", "wall", "ceiling",
                          "floor", "floor-wood", "pavement-merged", "road",
                          "ceiling-merged", "wall-other-merged"}

    def get_inst_mask(seg):
        seg_id = seg.get("id", None)
        if seg_id is not None:
            return (seg_map == seg_id).astype(np.uint8)
        x1, y1, x2, y2 = seg["bbox"]
        m = np.zeros((h, w), dtype=np.uint8)
        m[y1:y2, x1:x2] = 1
        return m

    for seg in segments:
        if seg["label"].lower() in BACKGROUND_LABELS:
            continue
        if seg.get("area", 0) < img_area * min_instance_area_ratio:
            continue
        instance_masks.append(get_inst_mask(seg))

    # ---------- u2net: 显著性连通区域 ----------
    sal_binary = (saliency_mask > saliency_thresh).astype(np.uint8)
    if sal_binary.sum() >= img_area * min_instance_area_ratio:
        num_labels, labels_im = cv2.connectedComponents(sal_binary)
        for label_id in range(1, num_labels):
            comp_mask = (labels_im == label_id).astype(np.uint8)
            if comp_mask.sum() >= img_area * min_instance_area_ratio:
                instance_masks.append(comp_mask)

    # ---------- depth: 最近的连通区域 (可选) ----------
    if depth_map is not None:
        depth_near = (depth_map > (1 - depth_near_ratio)).astype(np.uint8)
        if depth_near.sum() >= img_area * min_instance_area_ratio:
            num_labels, labels_im = cv2.connectedComponents(depth_near)
            for label_id in range(1, num_labels):
                comp_mask = (labels_im == label_id).astype(np.uint8)
                if comp_mask.sum() >= img_area * min_instance_area_ratio:
                    instance_masks.append(comp_mask)

    if len(instance_masks) == 0:
        # 兜底: 整图作为唯一"主体", 此时覆盖率约束相当于失效
        instance_masks.append(np.ones((h, w), dtype=np.uint8))

    return instance_masks


# ============================================================
# 去重 + 初筛
# ============================================================

def subject_coverage(box: BBox, instance_masks):
    """
    对每个实例 mask 分别计算 "框内该实例像素 / 该实例总像素",
    取最大值作为该框的主体覆盖率。

    这样无论真正的"主体"是 instance_masks 里的哪一个,
    只要框覆盖了其中某一个完整的实例, 覆盖率就会高。

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
        return 1.0, -1  # 没有任何有效实例, 不惩罚

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


def initial_filter(boxes, instance_masks, img_w, img_h,
                    coverage_thresh=SUBJECT_COVERAGE_THRESH,
                    min_area_ratio=MIN_AREA_RATIO,
                    max_area_ratio=MAX_AREA_RATIO,
                    keep_top_n=KEEP_TOP_N):
    """
    初筛策略调整: 不再追求"按score排序取TopN去逼近最优解",
    而是先做"温和的去除明显无效框", 再做"分层多样性采样"。

    动机:
    初筛阶段的目标只有两个 ——
      (a) 不要漏掉理想的最终框
      (b) 不要保留主体被严重截断/明显无效的框
    "哪个框最好"这个精细排序问题应交给最终评分阶段(可以用更复杂的
    美学/构图模型)。如果初筛阶段就用一个粗糙的几何分去排序+截断,
    很容易系统性地偏向某一类框(比如"小物体居中特写"), 把真正优秀
    但形态不同的候选框(全景/场景类构图)挤出100张之外。

    步骤:
    1. clip 边界, 计算 coverage 与 subject_idx (覆盖率仍用之前的定义,
       但只作为"是否严重截断"的依据, 不再主导排序)
    2. 宽松硬过滤: 去掉 coverage 过低(严重截断主体) 或
       面积比例过于极端(几乎不可裁 / 几乎是缩略图) 的框
    3. NMS 去重 (这一步的 score 仅用于"同组内选谁留下", 不影响最终
       是否保留这个"组")
    4. 分层多样性采样:
       - 第一层: 按 box.scale (候选框来源尺度, 0.2/0.3/.../0.6 等)分组
       - 第二层: 每个 scale 组内, 再按 subject_idx (覆盖的主体实例编号,
         -1 表示未匹配到任何实例/场景框) 分组
       每个 (scale, subject_idx) 组按 coverage 排序, 轮询(round-robin)
       从各组依次取框, 直到达到 keep_top_n。
       轮询保证: 不同尺度、不同"潜在主体"的框都有机会入选, 不会被
       单一维度的高分组垒满 100 个名额。

    返回:
        (final_records, all_records)
    """

    img_area = img_w * img_h

    records = []

    for box in boxes:
        b = clip_bbox(box, img_w, img_h)
        if b.width <= 0 or b.height <= 0:
            continue

        cov, subj_idx = subject_coverage(b, instance_masks)
        area_ratio = b.area / img_area

        records.append({
            "box": b,
            "coverage": cov,
            "subject_idx": subj_idx,
            "area_ratio": area_ratio,
        })

    # ---------- 1. 宽松硬过滤 ----------
    filtered = [
        r for r in records
        if r["coverage"] >= coverage_thresh
        and min_area_ratio <= r["area_ratio"] <= max_area_ratio
    ]

    # 过滤过头(比如全图都没有满足条件的框)则回退到全集
    if len(filtered) == 0:
        filtered = records

    # ---------- 2. NMS 去重 ----------
    # NMS 内部排序用 coverage 作为简单依据(覆盖率越高, 越"完整", 优先保留)
    boxes_f = [r["box"] for r in filtered]
    scores_f = [r["coverage"] for r in filtered]

    keep_idx = nms_dedup(boxes_f, scores_f, NMS_IOU_THRESH)
    deduped = [filtered[i] for i in keep_idx]

    # ---------- 3. 分层多样性采样 ----------
    from collections import defaultdict

    groups = defaultdict(list)
    for r in deduped:
        key = (r["box"].scale, r["subject_idx"])
        groups[key].append(r)

    # 每组内按 coverage 从高到低排序
    for key in groups:
        groups[key].sort(key=lambda r: r["coverage"], reverse=True)

    # 轮询从各组取框, 直到凑够 keep_top_n 或所有组耗尽
    group_keys = list(groups.keys())
    final = []
    pointer = {k: 0 for k in group_keys}

    while len(final) < keep_top_n:
        progressed = False
        for k in group_keys:
            if pointer[k] < len(groups[k]):
                final.append(groups[k][pointer[k]])
                pointer[k] += 1
                progressed = True
                if len(final) >= keep_top_n:
                    break
        if not progressed:
            break  # 所有组都已取完, 数量不足 keep_top_n 也只能这样

    # 为可视化方便, final 内部按 coverage 排个序(不影响"是否入选"的逻辑)
    final.sort(key=lambda r: r["coverage"], reverse=True)

    # 给每条记录补一个 "score" 字段(=coverage), 兼容后续可视化代码
    for r in final:
        r["score"] = r["coverage"]
    for r in records:
        if "score" not in r:
            r["score"] = r["coverage"]

    return final, records


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
        b = r["box"]
        crop = img_rgb[int(b.y1):int(b.y2), int(b.x1):int(b.x2)]
        crop = cv2.resize(crop, (160, 160))
        label = f"s={r['score']:.2f} c={r['coverage']:.2f} i={r['subject_idx']}"
        crop = cv2.copyMakeBorder(crop, 20, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(crop, label, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
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

    img_id = args.img
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
    instance_masks = build_instance_masks(img_rgb, saliency_mask, seg_map, segments, depth_map)
    print(f"  候选主体实例数: {len(instance_masks)}")
    for i, m in enumerate(instance_masks):
        print(f"    实例{i}: 面积占比={m.sum() / (h * w):.3f}")

    # ---------- 去重 + 初筛 ----------
    print("去重 + 初筛...")
    final_records, all_records = initial_filter(all_boxes, instance_masks, w, h)
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

    # 采样网格: 等间距抽取 20 张, 展示"多样性采样"后的整体分布效果
    # (而不是只看 coverage 最高的若干张, 因为现在的目标是多样性而非排序)
    n = len(final_records)
    if n <= 20:
        sample_records = final_records
    else:
        step = n / 20
        idxs = [int(i * step) for i in range(20)]
        sample_records = [final_records[i] for i in idxs]

    grid_img = draw_top_k(img_rgb, sample_records, k=20, framing_img=framing_img)
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
        f.write(f"覆盖率分布: min={coverages.min():.3f} "
                f"p25={np.percentile(coverages, 25):.3f} "
                f"median={np.median(coverages):.3f} "
                f"p75={np.percentile(coverages, 75):.3f} "
                f"max={coverages.max():.3f}\n\n")

        # 分层采样组成统计
        from collections import Counter
        scale_counter = Counter(r["box"].scale for r in final_records)
        subj_counter = Counter(r["subject_idx"] for r in final_records)
        f.write("最终100框按 scale 分布:\n")
        for k in sorted(scale_counter):
            f.write(f"  scale={k}: {scale_counter[k]}\n")
        f.write("最终100框按 subject_idx 分布 (-1=未匹配到任何实例):\n")
        for k in sorted(subj_counter, key=lambda x: (x is None, x)):
            f.write(f"  subject_idx={k}: {subj_counter[k]}\n")
        f.write("\n")

        f.write("等间距采样 20 框详情(对应 topk_grid.jpg):\n")
        for i, r in enumerate(sample_records):
            b = r["box"]
            f.write(f"{i+1:2d}. box=({b.x1},{b.y1},{b.x2},{b.y2}) "
                    f"scale={b.scale} coverage={r['coverage']:.3f} "
                    f"subject_idx={r['subject_idx']} "
                    f"area_ratio={r['area_ratio']:.3f}\n")

    print(f"\n完成! 输出目录: {OUTPUT_DIR}")
    print(f"  - {img_id}_overlay.jpg      所有保留框叠加图")
    print(f"  - {img_id}_topk_grid.jpg    Top20框裁剪拼接(含GT framing对比)")
    print(f"  - {img_id}_subject_mask.jpg 融合主体mask可视化")
    print(f"  - {img_id}_stats.txt        统计信息")


if __name__ == "__main__":
    main()'''