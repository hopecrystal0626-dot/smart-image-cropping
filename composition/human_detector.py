# composition/human_detector.py
import os

os.environ['MTCNN_HOME'] = r'D:\AI_Models\mtcnn_cache'


class HumanDetector:
    def __init__(self):
        self.detector = None
        self.available = False
        try:
            # 延迟导入，避免在环境不兼容时阻断主流程
            from mtcnn import MTCNN

            self.detector = MTCNN()
            self.available = True
            print("MTCNN initialized. Model cache: D:\\AI_Models\\mtcnn_cache")
        except Exception as e:
            print(f"MTCNN unavailable, fallback to empty human boxes: {e}")

    def detect_human_bboxes(self, image_rgb):
        """
        输入 RGB 图像，返回人脸边界框列表 [(x1,y1,x2,y2), ...]
        若 MTCNN 不可用，返回空列表。
        """
        if not self.available or self.detector is None:
            return []

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