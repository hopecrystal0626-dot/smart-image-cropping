"""
美学评分(CLIP ViT-L-14 + LAION Aesthetic Predictor) 单独测试脚本

复用 filter_and_visualize.py 的 initial_filter 产出的 100 个候选框,
仅用 CLIP 美学评分模型重新排序, 不混合任何规则分, 便于单独观察
CLIP 美学评分对当前候选框池的排序效果。

用法:
    python aesthetic_test.py --img A01
    python aesthetic_test.py --img A01,A02,A09
    python aesthetic_test.py --img all

输出 (data/output/aesthetic_test/):
    A01_aesthetic_grid.jpg   -- GT framing + 美学分Top20 裁剪拼接
    A01_aesthetic_stats.txt  -- 每个候选框的美学分明细(按分数排序)

依赖:
    pip install open_clip_torch --break-system-packages
    权重文件: weights/ava+logos-l14-linearMSE.pth
    (从 https://github.com/christophschuhmann/improved-aesthetic-predictor 获取)
"""

import os
import sys
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import open_clip
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# 复用初筛逻辑与候选框结构
from filter_and_visualize import (
    process_one,
    draw_top_k,
)


# ============================================================
# 配置
# ============================================================

OUTPUT_DIR = "data/output/aesthetic_test"
MODEL_PATH = "weights/ava+logos-l14-linearMSE.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# CLIP + Aesthetic Predictor
# ============================================================

class AestheticPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),

            nn.Linear(1024, 128),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


_clip_model = None
_preprocess = None
_predictor = None


def load_models():
    global _clip_model, _preprocess, _predictor

    if _clip_model is not None:
        return

    print("加载 CLIP ViT-L-14 ...")
    _clip_model, _, _preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="openai"
    )
    _clip_model = _clip_model.to(DEVICE)
    _clip_model.eval()

    print(f"加载美学评分头: {MODEL_PATH}")
    _predictor = AestheticPredictor()
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    _predictor.load_state_dict(state_dict)
    _predictor = _predictor.to(DEVICE)
    _predictor.eval()

    print("模型加载完成。\n")


def extract_feature(rgb_img):
    image = Image.fromarray(rgb_img)
    image_tensor = _preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feature = _clip_model.encode_image(image_tensor)
        feature = feature / feature.norm(dim=-1, keepdim=True)

    return feature


def aesthetic_score(rgb_img):
    feature = extract_feature(rgb_img)
    with torch.no_grad():
        score = _predictor(feature)
    return float(score.cpu().item())


def aesthetic_rerank(img_rgb, records):
    """
    对 records(initial_filter 输出的候选框记录列表)逐个裁剪打分,
    写入 record["aesthetic_score"], 并返回按该分数降序排序的新列表
    (不修改输入顺序, 返回一个排好序的副本列表)。
    """
    print(f"开始美学评分, 共 {len(records)} 个候选框...")

    h, w = img_rgb.shape[:2]

    for i, record in enumerate(records):
        box = record["box"]

        x1 = max(0, int(box.x1))
        y1 = max(0, int(box.y1))
        x2 = min(w, int(box.x2))
        y2 = min(h, int(box.y2))

        if x2 <= x1 or y2 <= y1:
            record["aesthetic_score"] = -999.0
            continue

        crop = img_rgb[y1:y2, x1:x2]

        if crop.size == 0:
            record["aesthetic_score"] = -999.0
            continue

        record["aesthetic_score"] = aesthetic_score(crop)

        if (i + 1) % 20 == 0 or (i + 1) == len(records):
            print(f"  {i + 1}/{len(records)}")

    ranked = sorted(records, key=lambda r: r["aesthetic_score"], reverse=True)
    return ranked


# ============================================================
# 可视化: 复用 draw_top_k, 但标签换成美学分
# ============================================================

def draw_aesthetic_top_k(img_rgb, records, k=20, framing_img=None):
    """
    与 filter_and_visualize.draw_top_k 类似, 但标签显示美学分,
    不区分主体/风景赛道(本测试只看纯 CLIP 美学排序结果)。
    """
    crops = []
    for r in records[:k]:
        b = r["box"]
        x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (160, 160))
        label = f"aes={r['aesthetic_score']:.2f} rule={r.get('score', 0):.2f}"
        crop = cv2.copyMakeBorder(crop, 20, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(crop, label, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
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
        row, col = divmod(i, cols)
        grid[row * 180: row * 180 + 180, col * 160: col * 160 + 160] = c

    return grid


# ============================================================
# 主流程
# ============================================================

def resolve_img_ids(img_arg, data_dir="data/testA"):
    if img_arg.lower() == "all":
        ids = []
        for fname in sorted(os.listdir(data_dir)):
            if not fname.lower().endswith(".jpg"):
                continue
            if "_framing" in fname:
                continue
            ids.append(os.path.splitext(fname)[0])
        return ids
    return [s.strip() for s in img_arg.split(",") if s.strip()]


def process_one_aesthetic(img_id, args):
    print(f"\n===== [美学评分] 处理 {img_id} =====")

    # 复用初筛流程, 得到 100 个候选框
    img_rgb, final_records, framing_img, instance_masks, landscape_masks, depth_map = process_one(img_id, args)

    # 美学评分重排序 (不修改 final_records 原顺序, 返回新排序列表)
    ranked = aesthetic_rerank(img_rgb, final_records)

    # 可视化: GT framing + 美学分 Top20
    grid_img = draw_aesthetic_top_k(img_rgb, ranked, k=20, framing_img=framing_img)
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"{img_id}_aesthetic_grid.jpg"),
        cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR)
    )

    # 统计文件: 全部100个框按美学分排序输出
    with open(os.path.join(OUTPUT_DIR, f"{img_id}_aesthetic_stats.txt"), "w", encoding="utf-8") as f:
        f.write(f"图片: {img_id}\n")
        f.write(f"候选框数量: {len(ranked)}\n\n")
        f.write("按美学分排序 (aes=CLIP美学分, rule=初筛阶段的score, "
                "L=是否风景赛道, i=subject_idx):\n")
        for idx, r in enumerate(ranked):
            b = r["box"]
            f.write(f"{idx+1:3d}. aes={r['aesthetic_score']:7.3f} "
                    f"rule={r['score']:.3f} L={'Y' if r['is_landscape'] else 'N'} "
                    f"i={r['subject_idx']:>3} "
                    f"box=({int(b.x1)},{int(b.y1)},{int(b.x2)},{int(b.y2)}) "
                    f"area_ratio={r['area_ratio']:.3f}\n")

    print(f"完成! 输出:")
    print(f"  - {img_id}_aesthetic_grid.jpg   美学分Top20裁剪拼接(含GT framing)")
    print(f"  - {img_id}_aesthetic_stats.txt  全部{len(ranked)}框美学分明细")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default="A01",
                         help="图片 ID, 支持单个/逗号分隔多个/'all'")
    parser.add_argument("--no_depth", action="store_true", help="跳过depth模型(加速调试)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    load_models()

    img_ids = resolve_img_ids(args.img)
    print(f"共 {len(img_ids)} 张图片待处理: {img_ids}")

    for img_id in img_ids:
        try:
            process_one_aesthetic(img_id, args)
        except Exception as e:
            print(f"[ERROR] 处理 {img_id} 失败: {e}")


if __name__ == "__main__":
    main()