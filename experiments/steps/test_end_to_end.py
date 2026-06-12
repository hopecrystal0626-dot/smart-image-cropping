from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from pipeline.config import PipelineConfig
from pipeline.runner import SystemizedSmartCropping


def main(image_path: str):
    config = PipelineConfig()
    config.fusion.top_k = 10
    runner = SystemizedSmartCropping(config)
    result = runner.run(image_path)
    top10 = result["top10"]
    print(f"image={Path(image_path).name}")
    print(f"top10={len(top10)}")
    if top10:
        print(f"best_rerank={top10[0]['rerank_score']:.4f}")
        print(f"best_fusion={top10[0]['fusion_score']:.4f}")
        print(f"best_bbox={(top10[0]['bbox'].x1, top10[0]['bbox'].y1, top10[0]['bbox'].x2, top10[0]['bbox'].y2)}")
        top5 = top10[:5]
        print("top5_bboxes=")
        for i, item in enumerate(top5, 1):
            bbox = item["bbox"]
            print(f"  #{i}: {(bbox.x1, bbox.y1, bbox.x2, bbox.y2)} score={item['rerank_score']:.4f}")

        top5_vis = runner.save_topk_visualization(image_path, top5, config.output_dir, k=5)
        print(f"top5_vis={top5_vis}")
    print(f"config={result['config']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="端到端系统化测试")
    parser.add_argument("image_path")
    args = parser.parse_args()
    main(args.image_path)
