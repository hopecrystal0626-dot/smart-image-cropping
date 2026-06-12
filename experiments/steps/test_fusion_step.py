from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

import cv2

from crop.candidate_generator import generate_candidates
from saliency.detector import FTDetector
from saliency.saliency_utils import filter_top_bboxes_by_percentile
from pipeline.config import PipelineConfig
from pipeline.runner import SystemizedSmartCropping


def main(image_path: str):
    config = PipelineConfig()
    runner = SystemizedSmartCropping(config)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    h, w = img.shape[:2]
    candidates = generate_candidates(w, h)
    sal_map = FTDetector().detect(img)
    top_bboxes, _, _ = filter_top_bboxes_by_percentile(sal_map, candidates, top_percent=config.saliency.top_percent)
    scored = runner.score_with_fusion(img, top_bboxes)
    print(f"image={Path(image_path).name}")
    print(f"top_bboxes={len(top_bboxes)}")
    print(f"scored={len(scored)}")
    if scored:
        best = scored[0]
        print(f"best_fusion={best['fusion_score']:.4f}")
        print(f"best_handcraft={best['handcraft_score']:.4f}")
        print(f"best_nima={best['nima_score']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="融合打分测试")
    parser.add_argument("image_path")
    args = parser.parse_args()
    main(args.image_path)
