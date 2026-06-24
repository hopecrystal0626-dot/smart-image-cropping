from pathlib import Path
import argparse
import os
import sys

sys.path.append(str(Path(__file__).parent))

from pipeline.config import PipelineConfig
from pipeline.runner import SystemizedSmartCropping


def resolve_image_path(raw_path: str) -> str:
	p = Path(raw_path)
	if p.exists():
		return str(p)

	data_root = Path("data")
	candidates = [
		data_root / raw_path,
		data_root / "testA" / raw_path,
		data_root / "demo" / raw_path,
	]

	if p.suffix == "":
		for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
			candidates.extend([
				data_root / f"{raw_path}{ext}",
				data_root / "testA" / f"{raw_path}{ext}",
				data_root / "demo" / f"{raw_path}{ext}",
			])

	for c in candidates:
		if c.exists():
			return str(c)

	for found in data_root.rglob("*"):
		if found.is_file() and (found.name == raw_path or found.stem == raw_path):
			return str(found)

	raise FileNotFoundError(f"未找到图片: {raw_path}。请传绝对路径，或把图片放到 data/ 下。")


def main():
	parser = argparse.ArgumentParser(description="智能取景系统统一入口")
	parser.add_argument("image_path", help="输入图片路径")
	parser.add_argument("--top_k", type=int, default=10, help="输出前K个候选框")
	parser.add_argument("--clip_mode", default="balanced", help="CLIP 评分模式")
	parser.add_argument("--output_dir", default="./data/output/final", help="输出目录")
	args = parser.parse_args()

	image_path = resolve_image_path(args.image_path)

	config = PipelineConfig()
	config.fusion.top_k = args.top_k
	config.fusion.clip_mode = args.clip_mode
	config.output_dir = args.output_dir

	runner = SystemizedSmartCropping(config)
	result = runner.run(image_path, top_k=args.top_k)
	top10 = result["top10"]

	print(f"image={Path(image_path).name}")
	print(f"resolved_path={image_path}")
	print(f"top10={len(top10)}")
	if top10:
		best = top10[0]
		print(f"best_rerank={best['rerank_score']:.4f}")
		print(f"best_fusion={best['fusion_score']:.4f}")
		print(f"best_bbox={(best['bbox'].x1, best['bbox'].y1, best['bbox'].x2, best['bbox'].y2)}")

		top5 = top10[:5]
		print("top5_bboxes=")
		for i, item in enumerate(top5, 1):
			bbox = item["bbox"]
			print(f"  #{i}: {(bbox.x1, bbox.y1, bbox.x2, bbox.y2)} score={item['rerank_score']:.4f}")

		os.makedirs(config.output_dir, exist_ok=True)
		crop_path, vis_path = runner.save_best_crop(image_path, best, config.output_dir)
		top5_vis_path = runner.save_topk_visualization(image_path, top5, config.output_dir, k=5)
		print(f"best_crop={crop_path}")
		print(f"best_vis={vis_path}")
		print(f"top5_vis={top5_vis_path}")


if __name__ == "__main__":
	main()

