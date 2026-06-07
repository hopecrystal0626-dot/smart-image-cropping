"""
批量测试加权融合版智能取景系统
测试 TestA 中的所有图片，输出每张图片的前10名候选框详情
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
from pathlib import Path

# 添加项目根目录到系统路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np

# 修改：从 complex/crop_system.py 导入
from complex.crop_system import SmartCropping
from experiments.test_saliency import extract_bbox_from_framing, compute_iou
from crop.bbox_utils import BBox


def test_single_image(img_path, framing_path, cropper):
    """测试单张图片，返回前10名和 IoU"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None, None, None
    
    # 获取老师框（用于计算 IoU）
    gt_bbox = None
    if framing_path.exists():
        bbox = extract_bbox_from_framing(str(img_path), str(framing_path))
        if bbox:
            gt_bbox = BBox(bbox[0], bbox[1], bbox[2], bbox[3], scale=1.0)
    
    # 获取前10个候选框
    try:
        results = cropper.get_top10_crops(str(img_path), top_k=10)
    except Exception as e:
        print(f"   ❌ 处理失败: {e}")
        return None, None, None, None
    
    if len(results) == 0:
        return None, None, None, None
    
    return results, gt_bbox, img


def visualize_top10(img, results, gt_bbox, output_path):
    """可视化前10个候选框"""
    canvas = img.copy()
    
    # 颜色列表
    colors = [
        (255, 0, 255),   # 第1名：粉色
        (0, 255, 0),     # 第2名：绿色
        (255, 255, 0),   # 第3名：青色
        (0, 165, 255),   # 第4名：橙色
        (255, 0, 0),     # 第5名：蓝色
        (128, 0, 128),   # 第6名：紫色
        (0, 128, 128),   # 第7名：深青
        (128, 128, 0),   # 第8名：橄榄
        (255, 128, 0),   # 第9名：橙黄
        (0, 128, 255)    # 第10名：天蓝
    ]
    
    # 老师框（白色粗线）
    if gt_bbox:
        cv2.rectangle(canvas, (gt_bbox.x1, gt_bbox.y1), (gt_bbox.x2, gt_bbox.y2), (255, 255, 255), 3)
        cv2.putText(canvas, "Teacher", (gt_bbox.x1, gt_bbox.y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # 画前10名框
    for i, item in enumerate(results):
        bbox = item['bbox']
        color = colors[i % len(colors)]
        thickness = 3 if i == 0 else 2
        cv2.rectangle(canvas, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, thickness)
        
        label = f"#{i+1}: F={item['fusion_score']:.3f}"
        cv2.putText(canvas, label, (bbox.x1, bbox.y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
    
    # 添加图例
    legend_y = 30
    cv2.putText(canvas, "Top 10 Candidates (Fusion Score)", (10, legend_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    for i, color in enumerate(colors[:5]):
        cv2.rectangle(canvas, (10, legend_y + 10 + i*25), (30, legend_y + 30 + i*25), color, -1)
        cv2.putText(canvas, f"#{i+1}", (35, legend_y + 25 + i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    cv2.imwrite(str(output_path), canvas)


def print_top10_table(img_name, results):
    """打印前10名表格"""
    print(f"\n📊 {img_name} 前10名详情:")
    print(f"{'排名':<6} {'融合分':<12} {'手工分':<12} {'NIMA分':<12} {'面积占比':<10}")
    print("-" * 55)
    
    for i, item in enumerate(results):
        print(f"#{i+1:<5} {item['fusion_score']:<12.4f} {item['handcraft_score']:<12.4f} "
              f"{item['nima_score']:<12.4f} {item['area_ratio']:<10.0%}")
    
    fusion_scores = [r['fusion_score'] for r in results]
    print(f"\n  最高融合分: {max(fusion_scores):.4f}")
    print(f"  最低融合分: {min(fusion_scores):.4f}")
    print(f"  平均融合分: {sum(fusion_scores)/len(fusion_scores):.4f}")


def save_each_crop(img, results, output_dir, img_name):
    """单独保存每个候选框的裁剪结果"""
    crop_dir = output_dir / f"{img_name}_crops"
    crop_dir.mkdir(exist_ok=True)
    
    for i, item in enumerate(results):
        bbox = item['bbox']
        cropped = img[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        if cropped.size > 0:
            fusion_score = item['fusion_score']
            crop_path = crop_dir / f"rank_{i+1}_fusion_{fusion_score:.3f}.jpg"
            cv2.imwrite(str(crop_path), cropped)


def main():
    testA_dir = project_root / "data/testA"
    output_dir = project_root / "data/output/batch_fusion_top10"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 融合权重
    ALPHA = 0.4  # 手工评分权重
    BETA = 0.6   # NIMA评分权重
    
    print("="*60)
    print("批量测试：加权融合版智能取景系统")
    print(f"融合权重: 手工评分={ALPHA}, NIMA评分={BETA}")
    print("输出：每张图片的前10名候选框详情")
    print("="*60)
    
    # 获取所有测试图片
    testA_path = Path(testA_dir)
    image_files = []
    for img_path in testA_path.glob("*.jpg"):
        if "_framing" in img_path.name:
            continue
        framing_path = testA_path / f"{img_path.stem}_framing.jpg"
        if framing_path.exists():
            image_files.append((img_path, framing_path))
    
    print(f"\n找到 {len(image_files)} 张测试图片\n")
    
    # 初始化系统
    cropper = SmartCropping(alpha=ALPHA, beta=BETA)
    
    # 全局统计
    all_fusion_scores = []
    all_handcraft_scores = []
    all_nima_scores = []
    
    for idx, (img_path, framing_path) in enumerate(image_files):
        img_name = img_path.stem
        print(f"\n[{idx+1}/{len(image_files)}] 处理: {img_name}.jpg")
        
        results, gt_bbox, img = test_single_image(img_path, framing_path, cropper)
        
        if results is None:
            print(f"   ❌ 处理失败")
            continue
        
        # 打印前10名表格
        print_top10_table(img_name, results)
        
        # 收集统计
        all_fusion_scores.extend([r['fusion_score'] for r in results])
        all_handcraft_scores.extend([r['handcraft_score'] for r in results])
        all_nima_scores.extend([r['nima_score'] for r in results])
        
        # 保存每个裁剪结果
        save_each_crop(img, results, output_dir, img_name)
        
        # 可视化前10名框
        vis_path = output_dir / f"{img_name}_top10_visualize.jpg"
        visualize_top10(img, results, gt_bbox, vis_path)
        print(f"   💾 可视化保存: {vis_path}")
    
    # 全局统计
    print("\n" + "="*60)
    print("全局统计（所有图片的前10名汇总）")
    print("="*60)
    
    if len(all_fusion_scores) > 0:
        print(f"\n总候选框数: {len(all_fusion_scores)}")
        print(f"平均融合分: {np.mean(all_fusion_scores):.4f}")
        print(f"平均手工分: {np.mean(all_handcraft_scores):.4f}")
        print(f"平均NIMA分: {np.mean(all_nima_scores):.4f}")
        print(f"最高融合分: {max(all_fusion_scores):.4f}")
        print(f"最低融合分: {min(all_fusion_scores):.4f}")
        
        # 按排名统计
        print("\n按排名统计（各位置的平均融合分）:")
        rank_scores = [[] for _ in range(10)]
        for i in range(0, len(all_fusion_scores), 10):
            for j in range(10):
                if i + j < len(all_fusion_scores):
                    rank_scores[j].append(all_fusion_scores[i + j])
        
        for j in range(10):
            if rank_scores[j]:
                avg = np.mean(rank_scores[j])
                print(f"  第{j+1}名: 平均融合分 = {avg:.4f}")
    
    print(f"\n✅ 测试完成！")
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    main()