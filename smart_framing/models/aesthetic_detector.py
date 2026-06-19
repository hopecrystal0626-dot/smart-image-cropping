# models/aesthetic_detector.py
import torch
import torch.nn as nn
import open_clip
from PIL import Image

class AestheticPredictor(nn.Module):
    """与权重文件 ava+logos-l14-linearMSE.pth 匹配的多层 MLP"""
    def __init__(self, input_dim=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


class AestheticDetector:
    def __init__(self, model_path="weights/ava+logos-l14-linearMSE.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP ViT-L-14 on {self.device}...")
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai', device=self.device
        )
        self.predictor = AestheticPredictor(768).to(self.device)
        print(f"Loading aesthetic predictor from {model_path}...")
        state_dict = torch.load(model_path, map_location=self.device)
        self.predictor.load_state_dict(state_dict)
        self.predictor.eval()
        print("AestheticDetector loaded successfully.")

    def predict_box_aesthetic(self, img_rgb, box):
        """计算某个特定候选框切片的美学分（保留原接口）"""
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return 4.0  # 异常安全分
        pil_img = Image.fromarray(crop)
        image_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            score = self.predictor(image_features).item()
        return score