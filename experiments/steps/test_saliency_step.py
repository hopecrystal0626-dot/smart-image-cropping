from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

import cv2

from crop.candidate_generator import generate_candidates
from saliency.detector import FTDetector
from saliency.saliency_utils import filter_top_bboxes_by_percentile


def main(image_path: str, top_percent: float = 0.3):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    h, w = img.shape[:2]
    candidates = generate_candidates(w, h)
    sal_map = FTDetector().detect(img)
    top_bboxes, top_scores, seg_ids = filter_top_bboxes_by_percentile(sal_map, candidates, top_percent=top_percent)
    print(f"image={Path(image_path).name}")
    print(f"top_percent={top_percent}")
    print(f"all_candidates={len(candidates)}")
    print(f"top_bboxes={len(top_bboxes)}")
    if top_scores:
        print(f"score_range=({min(top_scores):.4f}, {max(top_scores):.4f})")
    print(f"segment_hist={[seg_ids.count(i) for i in range(3)]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="显著性筛选测试")
    parser.add_argument("image_path")
    parser.add_argument("--top_percent", type=float, default=0.3)
    args = parser.parse_args()
    main(args.image_path, args.top_percent)
