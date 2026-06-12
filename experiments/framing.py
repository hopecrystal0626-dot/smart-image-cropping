import os
import json
import numpy as np
from PIL import Image
import torch
import cv2
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import ImageDraw


# ============================================================
# 模型加载
# ============================================================
print("正在加载 Mask2Former 模型...")
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-coco-panoptic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-coco-panoptic")
model.eval()
print("模型加载完成。\n")


PRIORITY = {
    1: 100, 17: 90, 18: 90, 19: 90, 20: 90, 21: 90, 22: 90, 23: 90, 24: 90, 25: 90,
    3: 70, 6: 70, 8: 70, 4: 70, 2: 70,
}


# ============================================================
# 主体定位 + 候选框 + 打分（与之前一致）
# ============================================================

def find_main_subject(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]

    segmentation = result["segmentation"].numpy()
    segments_info = result["segments_info"]
    if len(segments_info) == 0:
        return None

    best_seg_id, best_priority, best_area = None, -1, -1
    for seg in segments_info:
        priority = PRIORITY.get(seg["label_id"], 0)
        if priority <= 0:
            continue
        area = (segmentation == seg["id"]).sum()
        if priority > best_priority or (priority == best_priority and area > best_area):
            best_priority, best_area, best_seg_id = priority, area, seg["id"]

    if best_seg_id is None:
        return None

    if best_priority == 100:
        person_mask = np.zeros_like(segmentation, dtype=bool)
        for seg in segments_info:
            if PRIORITY.get(seg["label_id"], 0) == 100:
                person_mask |= (segmentation == seg["id"])
        return person_mask

    return segmentation == best_seg_id


def get_fallback_mask(image_size):
    W, H = image_size
    mask = np.zeros((H, W), dtype=bool)
    mask[H // 4: H * 3 // 4, W // 4: W * 3 // 4] = True
    return mask


def generate_candidates(image_size, mask_array):
    W, H = image_size
    ratio = W / H
    ys, xs = np.where(mask_array)
    if len(xs) == 0:
        return [(0, 0, W, H)]

    bx, by = int(xs.min()), int(ys.min())
    bw, bh = int(xs.max() - bx), int(ys.max() - by)
    cx, cy = bx + bw / 2, by + bh / 2

    candidates = []
    for scale in [0.6, 0.7, 0.8, 0.9, 1.0]:
        w = int(W * scale)
        h = int(round(w / ratio))
        if h > H:
            h = int(H * scale)
            w = int(round(h * ratio))
        x = int(cx - w / 2)
        y = int(cy - h / 2)
        x = max(0, min(x, W - w))
        y = max(0, min(y, H - h))
        candidates.append((x, y, w, h))

    return list(set(candidates))


def score_candidate(candidate, mask_array):
    x, y, w, h = candidate
    crop_mask = mask_array[y:y + h, x:x + w]
    subject_total = mask_array.sum()
    if subject_total == 0:
        return 0
    return crop_mask.sum() / subject_total


def get_best_framing(image_path):
    image = Image.open(image_path).convert("RGB")
    mask_array = find_main_subject(image)
    if mask_array is None:
        mask_array = get_fallback_mask(image.size)
    candidates = generate_candidates(image.size, mask_array)
    best_candidate = max(candidates, key=lambda c: score_candidate(c, mask_array))
    return best_candidate, mask_array


# ============================================================
# IoU 计算
# ============================================================

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2

    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1, area2 = w1 * h1, w2 * h2
    union = area1 + area2 - inter_area
    return inter_area / union


# ============================================================
# 关键新增：从 framing 裁剪图反推GT框坐标（模板匹配）
# ============================================================

def find_gt_box_by_template_matching(original_path, framing_path):
    """
    original_path: 原图路径，如 A01.jpg
    framing_path:  老师给的最优取景图，如 A01_framing.jpg
    返回: (x, y, w, h) —— framing图在原图中的位置和尺寸
    """
    original = cv2.imread(original_path)
    framing = cv2.imread(framing_path)

    oh, ow = original.shape[:2]
    fh, fw = framing.shape[:2]

    # 如果framing图比原图大，先按比例缩放到不超过原图（极少数情况）
    if fw > ow or fh > oh:
        scale = min(ow / fw, oh / fh)
        framing = cv2.resize(framing, (int(fw * scale), int(fh * scale)))
        fh, fw = framing.shape[:2]

    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    framing_gray = cv2.cvtColor(framing, cv2.COLOR_BGR2GRAY)

    # 模板匹配
    result = cv2.matchTemplate(original_gray, framing_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    x, y = max_loc
    return (x, y, fw, fh), max_val


# ============================================================
# 可视化
# ============================================================

def visualize_box(image_path, box, save_path, color="red", width=5):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    x, y, w, h = box
    draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
    image.save(save_path)


def visualize_both(image_path, pred_box, gt_box, save_path):
    """在同一张图上画预测框(红)和GT框(绿)，方便对比"""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    px, py, pw, ph = pred_box
    draw.rectangle([px, py, px + pw, py + ph], outline="red", width=5)

    gx, gy, gw, gh = gt_box
    draw.rectangle([gx, gy, gx + gw, gy + gh], outline="lime", width=5)

    image.save(save_path)


# ============================================================
# 批量测试 testA
# ============================================================

def run_test_on_dataset_A(test_dir="data/testA", num_images=20, save_vis=True, vis_dir="data/testA/vis"):
    if save_vis and not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    results = []
    total_iou = 0
    low_match_warnings = []

    for i in range(1, num_images + 1):
        idx = f"{i:02d}"  # 01, 02, ..., 20
        original_path = os.path.join(test_dir, f"A{idx}.jpg")
        framing_path = os.path.join(test_dir, f"A{idx}_framing.jpg")

        if not os.path.exists(original_path):
            print(f"警告：找不到 {original_path}，跳过")
            continue
        if not os.path.exists(framing_path):
            print(f"警告：找不到 {framing_path}，跳过")
            continue

        # 1. 反推GT框
        gt_box, match_val = find_gt_box_by_template_matching(original_path, framing_path)
        if match_val < 0.8:
            low_match_warnings.append((f"A{idx}", match_val))

        # 2. 预测框
        pred_box, mask_array = get_best_framing(original_path)

        # 3. IoU
        iou = compute_iou(pred_box, gt_box)
        total_iou += iou

        results.append({
            "image": f"A{idx}.jpg",
            "pred_box": pred_box,
            "gt_box": gt_box,
            "match_confidence": round(float(match_val), 4),
            "iou": round(iou, 4)
        })

        print(f"A{idx}: pred={pred_box}, gt={gt_box} (匹配度{match_val:.3f}), IoU={iou:.4f}")

        # 4. 可视化
        if save_vis:
            visualize_both(
                original_path, pred_box, gt_box,
                save_path=os.path.join(vis_dir, f"A{idx}_compare.jpg")
            )

    avg_iou = total_iou / len(results) if results else 0
    print(f"\n========== 测试完成 ==========")
    print(f"共测试 {len(results)} 张图片")
    print(f"平均 IoU: {avg_iou:.4f}")

    if low_match_warnings:
        print(f"\n警告：以下图片模板匹配置信度较低（<0.8），GT框可能不准确：")
        for name, val in low_match_warnings:
            print(f"  {name}: 匹配度={val:.3f}")

    results_sorted = sorted(results, key=lambda r: r["iou"])
    print(f"\nIoU最低的5张图片：")
    for r in results_sorted[:5]:
        print(f"  {r['image']}: IoU={r['iou']}, pred={r['pred_box']}, gt={r['gt_box']}")

    # 保存结果到json，方便后续分析
    with open(os.path.join(test_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n详细结果已保存到 {os.path.join(test_dir, 'eval_results.json')}")

    return results


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    run_test_on_dataset_A(test_dir="data/testA", num_images=20)