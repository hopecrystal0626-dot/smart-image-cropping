"""
智能取景系统 - 统一接口（加权融合版 + CLIP 平滑重排序）
流程：队友筛选候选框 → 融合评分 → 取前10 → CLIP 平滑分差 → 动态权重 → 重排序
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from composition.composition_score import compute_composition_score_single
from composition.aesthetic_scorer import AestheticScorer
from crop.candidate_generator import generate_candidates
from crop.bbox_utils import BBox
from saliency.detector import FTDetector
from saliency.saliency_utils import filter_top_bboxes_by_percentile
from clip_score import get_scorer


class SmartCropping:
    """智能取景系统主类（加权融合 + CLIP平滑重排序 + 动态权重）"""
    
    def __init__(self, yolo_model_path="yolov8n.pt", 
                 alpha=0.4, beta=0.6):
        self.yolo = YOLO(yolo_model_path)
        self.alpha = alpha  # 手工评分权重
        self.beta = beta    # NIMA评分权重
        
        self.nima_scorer = AestheticScorer()
        self.clip_scorer = get_scorer(mode="balanced")
        
        print(f"✅ 智能取景系统初始化完成")
        print(f"   融合权重: 手工={alpha}, NIMA={beta}")
        print(f"   CLIP平滑重排序: 动态压缩分差 + 动态权重 + 归一化范围[0.45,0.55]")
    
    def get_top10_crops(self, image_path, top_k=10, clip_mode="balanced"):
        """
        输入图片路径，返回前K个最优候选框
        流程：队友筛选 → 融合评分 → 取前10 → CLIP平滑分差 → 动态权重 → 重排序
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        self.clip_scorer.set_mode(clip_mode)
        
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # ========== 1. YOLO 物体检测 ==========
        all_objects = self._detect_objects(img)
        
        # ========== 2. 生成候选框 ==========
        all_candidates = generate_candidates(w, h)
        
        # ========== 3. 尺寸筛选 ==========
        MIN_AREA_RATIO = 0.10
        filtered_by_size = []
        for bbox in all_candidates:
            area_ratio = (bbox.width * bbox.height) / (w * h)
            if area_ratio >= MIN_AREA_RATIO:
                filtered_by_size.append(bbox)
        
        # ========== 4. 显著性筛选 ==========
        detector = FTDetector()
        sal_map = detector.detect(img)
        result = filter_top_bboxes_by_percentile(sal_map, filtered_by_size, 0.3)
        top_bboxes = result[0]
        
        # ========== 5. 完整性检测 ==========
        complete_boxes = []
        for bbox in top_bboxes:
            is_complete = self._check_completeness(bbox, all_objects)
            if len(all_objects) == 0 or is_complete:
                complete_boxes.append(bbox)
        
        if len(complete_boxes) == 0:
            complete_boxes = top_bboxes
        
        # ========== 6. 队友融合评分（手工 + NIMA）==========
        scored_boxes = []
        for bbox in complete_boxes:
            cropped = img[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
            if cropped.size == 0:
                continue
            
            score_detail = compute_composition_score_single(
                cropped, 
                bbox=(bbox.x1, bbox.y1, bbox.width, bbox.height)
            )
            handcraft_score = score_detail['total_score']
            
            try:
                nima_score = self.nima_scorer.get_score_for_bbox(img, bbox)
                nima_score = nima_score / 10.0
            except Exception as e:
                nima_score = 0.5
            
            fusion_score = self.alpha * handcraft_score + self.beta * nima_score
            
            scored_boxes.append({
                'bbox': bbox,
                'handcraft_score': handcraft_score,
                'nima_score': nima_score,
                'fusion_score': fusion_score,
                'area_ratio': (bbox.width * bbox.height) / (w * h)
            })
        
        # 先按融合分排序，取前10
        scored_boxes.sort(key=lambda x: x['fusion_score'], reverse=True)
        top10 = scored_boxes[:10]
        
        if len(top10) == 0:
            return []
        
        # ========== 7. CLIP 平滑重排序（分差压缩 + 动态权重）==========
        # 提取前10个框的图像
        cropped_images = []
        for item in top10:
            bbox = item['bbox']
            cropped = img_rgb[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
            cropped_images.append(cropped)
        
        # CLIP 批量评分
        clip_scores = self.clip_scorer.score_batch(cropped_images)
        
        # 平滑分差（动态压缩）
        clip_min, clip_max = min(clip_scores), max(clip_scores)
        clip_range = clip_max - clip_min
        
        # 根据分差决定压缩目标
        if clip_range > 0.04:
            target_range = 0.008
        elif clip_range > 0.025:
            target_range = 0.012
        elif clip_range > 0.015:
            target_range = 0.016
        else:
            target_range = clip_range
        
        if clip_range > target_range and clip_range > 0:
            compress_ratio = target_range / clip_range
            clip_scores_smoothed = [
                clip_min + (s - clip_min) * compress_ratio
                for s in clip_scores
            ]
            print(f"  🔧 CLIP 分差 {clip_range:.4f} → 压缩到 {target_range:.4f}")
        else:
            clip_scores_smoothed = clip_scores
        
        # 归一化 CLIP 分数到范围 [0.45, 0.55]（差距 0.1）
        min_s, max_s = min(clip_scores_smoothed), max(clip_scores_smoothed)
        if max_s > min_s:
            clip_scores_norm = [(s - min_s) / (max_s - min_s) for s in clip_scores_smoothed]
            clip_scores_norm = [0.45 + x * 0.1 for x in clip_scores_norm]
        else:
            clip_scores_norm = [0.50] * len(clip_scores)
        
        # 根据分差动态调整权重
        if clip_range > 0.035:
            clip_weight = 0.05
            fusion_weight = 0.95
            print(f"  ⚠️ CLIP 分差过大 ({clip_range:.4f})，权重降为 {clip_weight}")
        elif clip_range > 0.02:
            clip_weight = 0.10
            fusion_weight = 0.90
            print(f"  ⚠️ CLIP 分差中等 ({clip_range:.4f})，权重降为 {clip_weight}")
        elif clip_range > 0.01:
            clip_weight = 0.15
            fusion_weight = 0.85
        else:
            clip_weight = 0.20
            fusion_weight = 0.80
        
        # 计算重排分
        for i, item in enumerate(top10):
            item['clip_score_raw'] = clip_scores[i]
            item['clip_score_smoothed'] = clip_scores_smoothed[i]
            item['clip_score_norm'] = clip_scores_norm[i]
            item['clip_weight'] = clip_weight
            item['fusion_weight'] = fusion_weight
            item['rerank_score'] = clip_weight * clip_scores_norm[i] + fusion_weight * item['fusion_score']
        
        # 按重排分排序
        top10.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return top10[:top_k]
    
    def visualize_top5(self, image_path, results, save_path=None):
        """可视化前5个候选框"""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.imshow(img)
        ax.set_title(f"Top5 候选框（CLIP平滑重排序）", fontsize=14)
        ax.axis('off')
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, r in enumerate(results[:5]):
            bbox = r['bbox']
            color = colors[i % len(colors)]
            rect = patches.Rectangle(
                (bbox.x1, bbox.y1), bbox.width, bbox.height,
                linewidth=2.5, edgecolor=color, facecolor='none',
                label=f"#{i+1} (重排分={r['rerank_score']:.3f})"
            )
            ax.add_patch(rect)
            ax.text(
                bbox.x1, bbox.y1 - 8, f"#{i+1}",
                fontsize=12, color=color, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
            )
        
        ax.legend(loc='upper right', fontsize=10)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"    已保存对比图: {save_path.name}")
        plt.close()
    
    def save_best_crop(self, image_path, best_result, output_path):
        """保存最佳裁剪结果"""
        img = cv2.imread(image_path)
        bbox = best_result['bbox']
        cropped = img[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        cv2.imwrite(str(output_path), cropped)
        print(f"    已保存最佳裁剪图: {output_path.name}")
    
    def _detect_objects(self, img):
        """YOLO物体检测"""
        results = self.yolo(img)
        objects = []
        MIN_CONF = 0.3
        
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                if box.conf[0] < MIN_CONF:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                objects.append({
                    'bbox': (x1, y1, x2, y2),
                    'area': (x2 - x1) * (y2 - y1)
                })
        
        objects.sort(key=lambda x: x['area'], reverse=True)
        return objects
    
    def _check_completeness(self, candidate_bbox, all_objects, threshold=0.85):
        """检查候选框是否完整包含物体"""
        if len(all_objects) == 0:
            return True
        
        cx1, cy1, cx2, cy2 = candidate_bbox.x1, candidate_bbox.y1, candidate_bbox.x2, candidate_bbox.y2
        
        for obj in all_objects:
            ox1, oy1, ox2, oy2 = obj['bbox']
            ix1 = max(cx1, ox1)
            iy1 = max(cy1, oy1)
            ix2 = min(cx2, ox2)
            iy2 = min(cy2, oy2)
            
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                o_area = obj['area']
                completeness = inter_area / o_area
                if completeness >= threshold:
                    return True
        
        return False


# ========== 批量处理主程序 ==========
if __name__ == "__main__":
    output_dir = Path("data/output/clip_rerank_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    testA_dir = Path("data/testA")
    image_paths = sorted(testA_dir.glob("*.jpg"))
    image_paths = [p for p in image_paths if "_framing" not in p.name]
    
    print("=" * 60)
    print(f"CLIP平滑重排序 - 批量处理 {len(image_paths)} 张图片")
    print("=" * 60)
    
    cropper = SmartCropping(alpha=0.4, beta=0.6)
    
    for img_path in image_paths:
        img_name = img_path.stem
        
        print(f"\n处理: {img_name}.jpg")
        print("-" * 40)
        
        try:
            results = cropper.get_top10_crops(str(img_path), top_k=10, clip_mode="balanced")
            
            if not results:
                print(f"  ⚠️ 未找到候选框")
                continue
            
            print(f"  {'排名':<4} {'融合分':<10} {'CLIP原始':<10} {'CLIP平滑':<10} {'CLIP归一':<10} {'重排分':<10} {'面积':<8}")
            print(f"  {'-'*80}")
            for i, r in enumerate(results):
                print(f"  #{i+1:<3} {r['fusion_score']:<10.4f} {r['clip_score_raw']:<10.4f} "
                      f"{r.get('clip_score_smoothed', r['clip_score_raw']):<10.4f} {r['clip_score_norm']:<10.4f} "
                      f"{r['rerank_score']:<10.4f} {r['area_ratio']:<8.0%}")
            print(f"  📊 CLIP权重={r.get('clip_weight', 0.2):.2f}, 融合分权重={r.get('fusion_weight', 0.8):.2f}")
            
            best_result = results[0]
            best_crop_path = output_dir / f"{img_name}_best_crop.jpg"
            cropper.save_best_crop(str(img_path), best_result, best_crop_path)
            
            top5_viz_path = output_dir / f"{img_name}_top5_comparison.png"
            cropper.visualize_top5(str(img_path), results, save_path=top5_viz_path)
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ 批量处理完成！")
    print(f"📁 结果保存在: {output_dir}")
    print("=" * 60)