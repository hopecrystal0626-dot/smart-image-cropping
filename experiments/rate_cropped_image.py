"""
快速测试：对比自定义裁剪框、老师裁剪图和老师原图的得分
"""

import cv2
import sys
import os

# 添加项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from composition.composition_score import compute_composition_score_single


def rate_full_image(img, name):
    """给整张图片评分"""
    if img is None:
        print(f"{name}: 无法读取")
        return None
    
    h, w = img.shape[:2]
    r = compute_composition_score_single(img, (0, 0, w, h))
    print(f"{name}: 综合得分 {r['total_score']:.4f} (三分法:{r['thirds_score']:.3f} 平衡度:{r['balance_score']:.3f} 留白:{r['whitespace_score']:.3f})")
    return r['total_score']


def rate_bbox_on_image(img, bbox, name):
    """给原图上的某个区域评分（不裁剪，直接计算该区域的构图）"""
    if img is None:
        print(f"{name}: 无法读取")
        return None
    
    x, y, w, h = bbox
    r = compute_composition_score_single(img, (x, y, w, h))
    print(f"{name}: 综合得分 {r['total_score']:.4f} (三分法:{r['thirds_score']:.3f} 平衡度:{r['balance_score']:.3f} 留白:{r['whitespace_score']:.3f})")
    return r['total_score']


# ========== 修改这里的路径 ==========
original_path = "D:/VSCODE project/jiqishijue/smart-image-cropping/data/testA/A01.jpg"
teacher_path = "D:/VSCODE project/jiqishijue/smart-image-cropping/data/testA/A01_framing.jpg"

# ========== 定义原图上的候选框坐标（x, y, w, h, 名称）==========
# 这些坐标是相对于原图 765x342 的
my_boxes = [
    (100, 50, 300, 200, "我的框1:偏左"),
    (365, 50, 300, 200, "我的框2:偏右"),
    (150, 60, 400, 200, "我的框3:居中偏左"),
    (200, 50, 350, 220, "我的框4:适中"),
]
# ===================================

print("="*50)
print("构图评分测试")
print("="*50)

# 1. 老师原图（裁剪前）- 整张图评分
original_img = cv2.imread(original_path)
if original_img is not None:
    rate_full_image(original_img, "老师原图(裁剪前)")
else:
    print("老师原图读取失败")

# 2. 老师裁剪图 - 整张图评分
teacher_img = cv2.imread(teacher_path)
if teacher_img is not None:
    rate_full_image(teacher_img, "老师裁剪图")
else:
    print("老师裁剪图读取失败")

# 3. 原图上的自定义框 - 直接在原图上评分
if original_img is None:
    print("原图读取失败")
    sys.exit()

print("\n原图上的自定义裁剪框:")
for x, y, w, h, name in my_boxes:
    # 确保坐标在图像范围内
    x = max(0, min(x, original_img.shape[1] - 1))
    y = max(0, min(y, original_img.shape[0] - 1))
    w = min(w, original_img.shape[1] - x)
    h = min(h, original_img.shape[0] - y)
    rate_bbox_on_image(original_img, (x, y, w, h), name)

print("\n✅ 测试完成")