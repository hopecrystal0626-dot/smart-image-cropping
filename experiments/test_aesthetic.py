import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import matplotlib.pyplot as plt

from crop.candidate_generator import generate_candidates
from crop.bbox_utils import BBox
from saliency.saliency_utils import extract_bbox_from_framing
from composition.scoring import AestheticPipeline

def load_testA_pairs(testA_dir):
    pairs = []
    testA_path = Path(testA_dir)
    for img_path in testA_path.glob("*.jpg"):
        if "_framing" in img_path.name:
            continue
        framing_path = testA_path / f"{img_path.stem}_framing.jpg"
        if not framing_path.exists():
            continue
        bbox = extract_bbox_from_framing(str(img_path), str(framing_path))
        if bbox is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gt_bbox = BBox(bbox[0], bbox[1], bbox[2], bbox[3], scale=1.0)
        pairs.append((img_path.name, img, gt_bbox))
    return pairs

def main():
    testA_dir = "./data/testA"
    output_dir = "./data/output/final_pipeline"
    os.makedirs(output_dir, exist_ok=True)

    pairs = load_testA_pairs(testA_dir)
    print(f"成功加载 {len(pairs)} 张图像")

    # 初始化 pipeline（可使用默认参数，也可自定义）
    pipeline = AestheticPipeline()

    for img_name, img, gt_bbox in pairs:
        h, w = img.shape[:2]
        candidates = generate_candidates(w, h)
        if not candidates:
            continue

        final_bboxes, final_scores, human_boxes, object_boxes, sal_map = pipeline.process(img, candidates)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"{img_name} (top {len(final_bboxes)} boxes)")
        axes[0].axis('off')
        # GT 绿色框
        gt_rect = plt.Rectangle((gt_bbox.x1, gt_bbox.y1), gt_bbox.width, gt_bbox.height,
                                fill=False, edgecolor='green', lw=2, label='GT')
        axes[0].add_patch(gt_rect)
        # 人体框（蓝色）
        for hb in human_boxes:
            axes[0].add_patch(plt.Rectangle((hb[0], hb[1]), hb[2]-hb[0], hb[3]-hb[1],
                                            fill=False, edgecolor='blue', lw=1.5, label='Human'))
        # 物体框（青色）
        for ob in object_boxes:
            axes[0].add_patch(plt.Rectangle((ob[0], ob[1]), ob[2]-ob[0], ob[3]-ob[1],
                                            fill=False, edgecolor='cyan', lw=1.5, label='Object'))
        # 最终框
        for idx, (bbox, score) in enumerate(zip(final_bboxes, final_scores)):
            color = 'yellow' if idx == 0 else 'red'
            lw = 3 if idx == 0 else 1.5
            label = f'Best ({score:.2f})' if idx == 0 else (f'Top{idx+1}' if idx < 3 else '')
            axes[0].add_patch(plt.Rectangle((bbox.x1, bbox.y1), bbox.width, bbox.height,
                                            fill=False, edgecolor=color, lw=lw, label=label))
        axes[0].legend(loc='upper right', fontsize=8)

        axes[1].imshow(sal_map, cmap='hot')
        axes[1].set_title("Saliency Map")
        axes[1].axis('off')

        out_path = os.path.join(output_dir, f"{img_name}_final.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"{img_name}: {len(final_bboxes)} boxes")

    print(f"完成，结果保存在 {output_dir}")

if __name__ == "__main__":
    main()