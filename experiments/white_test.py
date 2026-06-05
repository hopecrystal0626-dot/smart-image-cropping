"""
测试留白模块
"""

import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from composition.whitespace import compute_whitespace_score


def test_whitespace():
    # 读取图片
    img_path = "D:/VSCODE project/jiqishijue/smart-image-cropping/data/testA/A01_framing.jpg"
    img = cv2.imread(img_path)
    
    if img is None:
        print("图片读取失败")
        return
    
    h, w = img.shape[:2]
    bbox = (0, 0, w, h)
    
    # 开启调试
    score = compute_whitespace_score(img, bbox, debug=True)
    print(f"\n最终留白得分: {score:.4f}")


if __name__ == "__main__":
    test_whitespace()