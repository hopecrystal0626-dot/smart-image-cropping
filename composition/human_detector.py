# composition/human_detector.py
import os
os.environ['MTCNN_HOME'] = r'D:\AI_Models\mtcnn_cache'  # 模型缓存到D盘

from mtcnn import MTCNN
import cv2

class HumanDetector:
    def __init__(self):
        self.detector = MTCNN()
        print("MTCNN (mtcnn library) initialized. Model cache: D:\\AI_Models\\mtcnn_cache")

    def detect_human_bboxes(self, image_rgb):
        """
        输入 RGB 图像，返回人脸边界框列表 [(x1,y1,x2,y2), ...]
        """
        h, w, _ = image_rgb.shape
        results = self.detector.detect_faces(image_rgb)
        boxes = []
        for res in results:
            x, y, width, height = res['box']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + width)
            y2 = min(h, y + height)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
        return boxes