import cv2
import numpy as np
from typing import Tuple, Optional

def extract_bbox_from_framing(original_img_path: str, framing_img_path: str) -> Optional[Tuple[int, int, int, int]]:
    """
    从原图和裁剪结果图（framing.jpg）中计算出取景框坐标
    返回 (x1, y1, x2, y2)
    """
    original = cv2.imread(original_img_path)
    framing = cv2.imread(framing_img_path)
    if original is None or framing is None:
        return None
    
    # 方法1：模板匹配（简单，当裁剪图是原图严格子区域时有效）
    result = cv2.matchTemplate(original, framing, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val > 0.8:   # 匹配度高
        x, y = max_loc
        h, w = framing.shape[:2]
        return (x, y, x + w, y + h)
    
    # 方法2：如果模板匹配失败（例如裁剪图有轻微缩放或颜色变化），用边缘检测+轮廓
    # 这里简单返回None，实际你可根据需要扩展
    return None

# 测试：遍历testA目录，自动生成每个原图的bbox坐标
if __name__ == "__main__":
    import glob
    testA_dir = "./data/testA"
    for orig_path in glob.glob(f"{testA_dir}/*.jpg"):
        if "_framing" in orig_path:
            continue
        framing_path = orig_path.replace(".jpg", "_framing.jpg")
        bbox = extract_bbox_from_framing(orig_path, framing_path)
        if bbox:
            print(f"{orig_path} -> {bbox}")
        else:
            print(f"Failed: {orig_path}")