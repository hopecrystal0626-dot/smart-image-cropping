import os
import urllib.request
from ultralytics import YOLO

MODEL_DIR = "D:/AI_Models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n-seg.pt")

if not os.path.exists(MODEL_PATH):
    print("YOLOv8n-seg 模型不存在，正在从镜像下载...")
    mirror_url = "https://github.moeyy.xyz/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-seg.pt"
    try:
        urllib.request.urlretrieve(mirror_url, MODEL_PATH)
        print("模型下载完成")
    except Exception as e:
        print(f"自动下载失败: {e}")
        print(f"请手动下载模型文件并放置到: {MODEL_PATH}")

class ObjectDetector:
    def __init__(self, conf_threshold=0.25):
        self.conf_threshold = conf_threshold
        self.model = YOLO(MODEL_PATH)
        print(f"YOLOv8 model loaded from {MODEL_PATH}, threshold={conf_threshold}")

    def detect_all(self, image_rgb, verbose=False):
        """
        一次推理同时获取人体框和其他物体框
        返回: (human_boxes, object_boxes)
        """
        results = self.model(image_rgb, conf=self.conf_threshold, verbose=False)
        human_boxes = []
        object_boxes = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            for box, cls in zip(boxes, cls_ids):
                if cls == 0:  # 人的类别
                    human_boxes.append((box[0], box[1], box[2], box[3]))
                else:
                    object_boxes.append((box[0], box[1], box[2], box[3]))
        if verbose:
            print(f"    一次推理: 人体 {len(human_boxes)} 个, 物体 {len(object_boxes)} 个")
        return human_boxes, object_boxes

    # 以下方法保留但内部调用 detect_all 以避免重复推理（可选）
    def detect_humans(self, image_rgb, verbose=False):
        human_boxes, _ = self.detect_all(image_rgb, verbose)
        return human_boxes

    def detect_objects(self, image_rgb, verbose=False):
        _, object_boxes = self.detect_all(image_rgb, verbose)
        return object_boxes