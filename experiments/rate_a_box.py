"""
集成同学A的候选框生成 + 同学C的构图评分（对所有框评分）
"""

import cv2
import sys
import os

# 添加项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from composition.composition_score import compute_composition_score_single
from crop.candidate_generator import generate_candidates


def box_to_bbox(box):
    """
    将A同学的BBox对象转换为 (x, y, w, h) 格式
    
    A同学的BBox属性: x1, y1, x2, y2
    """
    x = box.x1
    y = box.y1
    w = box.x2 - box.x1
    h = box.y2 - box.y1
    return (x, y, w, h)


def rate_single_box(img, bbox, name=""):
    """给单个框评分并打印"""
    x, y, w, h = bbox
    result = compute_composition_score_single(img, (x, y, w, h))
    
    print(f"{name}:")
    print(f"  位置: ({x}, {y}, {w}, {h})")
    print(f"  三分法: {result['thirds_score']:.4f}")
    print(f"  平衡度: {result['balance_score']:.4f}")
    print(f"  留白: {result['whitespace_score']:.4f}")
    print(f"  综合得分: {result['total_score']:.4f}")
    print()
    
    return result['total_score']


def main():
    # ========== 配置 ==========
    image_path = r"D:\VSCODE project\jiqishijue\smart-image-cropping\data\testA\A15.jpg"
    
    # ========== 1. 读取图片 ==========
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 图片读取失败: {image_path}")
        return
    
    h, w = img.shape[:2]
    print(f"📷 图片尺寸: {w} x {h}")
    print("="*50)
    
    # ========== 2. 调用A同学生成候选框 ==========
    print("\n🔍 正在生成候选框...")
    boxes = generate_candidates(img_w=w, img_h=h)
    print(f"✅ A同学生成了 {len(boxes)} 个候选框")
    print(f"📋 将对全部 {len(boxes)} 个框进行评分\n")
    
    # ========== 3. 对所有框评分，记录最佳 ==========
    print("="*50)
    print("🎯 构图评分中...")
    print("="*50 + "\n")
    
    best_score = -1
    best_index = -1
    best_bbox = None
    
    for i, box in enumerate(boxes):
        bbox = box_to_bbox(box)
        score = compute_composition_score_single(img, bbox)['total_score']
        
        # 打印进度（每100个框打印一次）
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  已处理: {i+1}/{len(boxes)} 个框, 当前最佳得分: {best_score:.4f}")
        
        if score > best_score:
            best_score = score
            best_index = i
            best_bbox = bbox
    
    # ========== 4. 输出最佳结果 ==========
    print("\n" + "="*50)
    print("🏆 最佳候选框结果")
    print("="*50)
    print(f"最佳框索引: 第 {best_index + 1} 个")
    print(f"最佳框位置: ({best_bbox[0]}, {best_bbox[1]}, {best_bbox[2]}, {best_bbox[3]})")
    print(f"最佳综合得分: {best_score:.4f}")
    
    # 详细输出最佳框的各维度得分
    result = compute_composition_score_single(img, best_bbox)
    print(f"\n📊 最佳框详细得分:")
    print(f"  三分法: {result['thirds_score']:.4f}")
    print(f"  平衡度: {result['balance_score']:.4f}")
    print(f"  留白: {result['whitespace_score']:.4f}")
    print("="*50)
    
    # ========== 5. 可视化最佳框 ==========
    show_vis = input("\n是否显示可视化？(y/n): ")
    if show_vis.lower() == 'y':
        canvas = img.copy()
        
        # 绘制最佳框（绿色加粗）
        x, y, w_box, h_box = best_bbox
        cv2.rectangle(canvas, (x, y), (x + w_box, y + h_box), (0, 255, 0), 3)
        cv2.putText(canvas, f"BEST SCORE: {best_score:.4f}", 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 显示
        cv2.imshow("Best Candidate Box", cv2.resize(canvas, (800, 600)))
        print("\n按任意键关闭图片窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()