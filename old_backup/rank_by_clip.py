"""
使用 CLIP 对队友筛选后的候选框进行评分排序
适配方案三：加权投票机制（支持模式切换）
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crop.candidate_generator import generate_candidates
from crop.bbox_utils import BBox
from saliency.saliency_utils import (
    get_saliency_map,
    filter_top_bboxes_by_percentile,
)
from clip_score import get_scorer, set_global_mode, list_available_modes, get_current_mode


def get_cropped_region(img, bbox):
    """从图像中裁剪出候选框区域"""
    return img[bbox.y1:bbox.y2, bbox.x1:bbox.x2]


def visualize_all_candidates(
    img_rgb: np.ndarray,
    candidates: List[BBox],
    saliency_map: np.ndarray,
    output_dir: str,
    img_name: str
):
    """可视化所有候选框"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"所有候选框 ({len(candidates)}个)", fontsize=14)
    axes[0].axis('off')
    
    for i, bbox in enumerate(candidates):
        if i < 5:
            color = 'red'
            lw = 2
        elif i < 20:
            color = 'orange'
            lw = 1.5
        elif i < 50:
            color = 'green'
            lw = 1
        else:
            color = 'gray'
            lw = 0.5
        rect = Rectangle(
            (bbox.x1, bbox.y1), bbox.width, bbox.height,
            fill=False, edgecolor=color, linewidth=lw
        )
        axes[0].add_patch(rect)
    
    axes[1].imshow(img_rgb)
    axes[1].imshow(saliency_map, cmap='hot', alpha=0.4)
    axes[1].set_title("显著性热力图", fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{img_name}_all_candidates.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存候选框分布图: {save_path}")


def score_candidates_with_clip(
    img_path: str,
    candidates: List[BBox],
    use_rgb: bool = True,
    mode: Optional[str] = None
) -> List[Tuple[BBox, float]]:
    """使用 CLIP 对候选框进行评分"""
    img = cv2.imread(img_path)
    if use_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    total = len(candidates)
    print(f"  开始 CLIP 评分 ({total} 个候选框)...")
    
    print("  步骤1: 准备候选框图像...")
    cropped_images = []
    for i, bbox in enumerate(candidates):
        cropped = get_cropped_region(img, bbox)
        cropped_images.append(cropped)
        if (i + 1) % 100 == 0:
            print(f"      进度: {i+1}/{total}")
    
    print("  步骤2: 批量评分...")
    scorer = get_scorer()
    
    if mode:
        scorer.set_mode(mode)
        print(f"  ✅ 已切换模式: {mode}")
    
    raw_scores = scorer.score_batch(cropped_images)
    
    results = [(bbox, float(score)) for bbox, score in zip(candidates, raw_scores)]
    results.sort(key=lambda x: x[1], reverse=True)

    if raw_scores:
        print(f"    分数范围: {min(raw_scores):.4f} -> {max(raw_scores):.4f}")
        print(f"    分数均值: {np.mean(raw_scores):.4f}")

    return results


def process_single_image(
    img_path: str,
    output_dir: str,
    top_percent: float = 0.5,
    clip_top_n: int = 5,
    mode: Optional[str] = None
):
    """处理单张图片"""
    print(f"\n{'='*60}")
    print(f"处理: {Path(img_path).name}")
    if mode:
        print(f"评分模式: {mode}")
    print(f"{'='*60}")

    img = cv2.imread(str(img_path))
    if img is None:
        print("  无法读取图片")
        return None

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("  生成候选框...")
    all_candidates = generate_candidates(w, h)
    print(f"  生成 {len(all_candidates)} 个候选框")

    print(f"  显著性筛选 (保留前 {top_percent*100}%)...")
    saliency_map = get_saliency_map(img)
    top_bboxes, _, _ = filter_top_bboxes_by_percentile(
        saliency_map, all_candidates, top_percent, num_segments=3
    )
    print(f"  筛选后剩余 {len(top_bboxes)} 个候选框")

    # 只保留前50个框
    if len(top_bboxes) > 50:
        top_bboxes = top_bboxes[:50]
        print(f"  📌 截断后剩余 50 个候选框")

    if not top_bboxes:
        print("  没有候选框")
        return None

    visualize_all_candidates(img_rgb, top_bboxes, saliency_map, output_dir, Path(img_path).stem)

    scored = score_candidates_with_clip(str(img_path), top_bboxes, mode=mode)

    print(f"\n  CLIP 评分排名 (前 {clip_top_n} 名):")
    for i, (bbox, score) in enumerate(scored[:clip_top_n], 1):
        print(f"    {i}. score={score:.4f} | "
              f"({bbox.x1},{bbox.y1},{bbox.x2},{bbox.y2})")

    visualize_results(img_rgb, scored, output_dir, Path(img_path).stem, clip_top_n)
    return scored


def visualize_results(
    img_rgb: np.ndarray,
    scored: List[Tuple[BBox, float]],
    output_dir: str,
    img_name: str,
    top_n: int = 5
):
    """可视化排名结果"""
    os.makedirs(output_dir, exist_ok=True)
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(img_rgb)
    
    current_mode = get_current_mode()
    ax.set_title(f"CLIP 评分排名 - {current_mode}", fontsize=14)
    ax.axis('off')

    for i, (bbox, score) in enumerate(scored[:top_n]):
        color = colors[i % len(colors)]
        rect = Rectangle(
            (bbox.x1, bbox.y1), bbox.width, bbox.height,
            fill=False, edgecolor=color, linewidth=2.5,
            label=f"#{i+1} (score={score:.3f})"
        )
        ax.add_patch(rect)
        ax.text(
            bbox.x1, bbox.y1 - 5, f"#{i+1} {score:.3f}",
            fontsize=10, color=color, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7)
        )

    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{img_name}_clip_rankings.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def batch_process(
    testA_dir: str, 
    output_dir: str, 
    top_percent: float = 0.5, 
    limit: Optional[int] = None,
    mode: Optional[str] = None
):
    """批量处理"""
    os.makedirs(output_dir, exist_ok=True)
    images = list(Path(testA_dir).glob("*.jpg"))
    images = [img for img in images if '_framing' not in img.name]

    if limit:
        images = images[:limit]

    print(f"\n找到 {len(images)} 张图片")
    print(f"显著性筛选比例: {top_percent*100}%")
    if mode:
        print(f"CLIP 评分模式: {mode}")
    print("=" * 60)

    for i, img_path in enumerate(images):
        print(f"\n进度: {i+1}/{len(images)}")
        process_single_image(str(img_path), output_dir, top_percent, clip_top_n=5, mode=mode)


def interactive_mode():
    """交互式模式选择"""
    print("\n" + "="*60)
    print("🎨 CLIP 美学评分器 - 方案三（加权投票机制）")
    print("="*60)
    
    scorer = get_scorer()
    modes = scorer.list_modes()
    
    print("\n可用模式:")
    for i, (mode_key, mode_info) in enumerate(modes.items(), 1):
        print(f"  {i}. {mode_info['name']}")
        print(f"     {mode_info['desc']}")
    
    print(f"\n当前模式: {scorer.get_mode_info()['name']}")
    print("\n选择模式 (输入数字或名称):")
    print("  - 输入数字 (1-5)")
    print("  - 输入名称 (balanced/portrait/landscape/vibrant/soft)")
    print("  - 直接回车保持当前模式")
    
    try:
        choice = input("\n请选择: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  保持当前模式")
        return None
    
    if not choice:
        print(f"  保持当前模式: {scorer.get_mode_info()['name']}")
        return None
    
    if choice.isdigit():
        idx = int(choice) - 1
        mode_keys = list(modes.keys())
        if 0 <= idx < len(mode_keys):
            mode = mode_keys[idx]
            scorer.set_mode(mode)
            return mode
    
    if choice in modes:
        scorer.set_mode(choice)
        return choice
    
    print(f"  ❌ 无效选择: {choice}")
    return None


def main():
    """主函数"""
    testA_dir = "data/testA"
    output_dir = "data/output/clip_ranking_results"

    if not os.path.exists(testA_dir):
        print(f"错误: 目录不存在 {testA_dir}")
        return
    
    # 交互式选择模式
    selected_mode = interactive_mode()
    
    # 询问是否批量处理
    print("\n" + "-"*60)
    try:
        run_batch = input("是否开始批量处理? (y/n, 默认 y): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        run_batch = 'n'
    
    if run_batch == 'n':
        print("退出程序")
        return
    
    # 获取当前模式
    current_mode = get_current_mode()
    
    # 批量处理
    batch_process(testA_dir, output_dir, top_percent=0.5, limit=20, mode=current_mode)
    
    print("\n" + "="*60)
    print("🎉 全部处理完成！")
    print(f"📁 结果保存在: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
