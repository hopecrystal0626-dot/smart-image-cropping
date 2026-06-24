from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

import cv2

from crop.candidate_generator import generate_candidates


def main(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    h, w = img.shape[:2]
    candidates = generate_candidates(w, h)
    print(f"image={Path(image_path).name}")
    print(f"size={w}x{h}")
    print(f"candidates={len(candidates)}")
    print(f"first5={[ (b.x1, b.y1, b.x2, b.y2, b.scale) for b in candidates[:5] ]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="候选框生成测试")
    parser.add_argument("image_path")
    args = parser.parse_args()
    main(args.image_path)
