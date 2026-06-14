import torch
import cv2
import numpy as np
from saliency.models.u2net import U2NET
from collections import OrderedDict

MODEL_PATH = r"D:\AI_Models\u2net.pth"

class U2NetDetector:
    _model = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")

        if U2NetDetector._model is None:
            print("正在加载 U2Net 权重...")
            net = U2NET(3, 1)
            
            # 安全加载权重，防止多卡训练带来的命名冲突
            state_dict = torch.load(MODEL_PATH, map_location=self.device)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
                
            net.load_state_dict(new_state_dict, strict=False)
            net.to(self.device)
            net.eval()
            U2NetDetector._model = net
            print("U2Net 权重加载成功！")

        self.model = U2NetDetector._model

    def predict(self, image_rgb):
        h, w = image_rgb.shape[:2]
        
        # 1. 严格的官方标准输入预处理
        img = cv2.resize(image_rgb, (320, 320))
        img = img.astype(np.float32) / 255.0
        
        # ImageNet 标准归一化（不加这个模型输出会极度不稳定）
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        img = img.transpose(2, 0, 1)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 2. 模型推理与激活
        with torch.no_grad():
            d0, d1, d2, d3, d4, d5, d6 = self.model(img)
            mix_tensor = d1
            pred_tensor = mix_tensor[0, 0, :, :].squeeze()

        pred = pred_tensor.cpu().numpy()

        # 3. 【核心修复】全局最大最小值拉伸
        # 这一步能强行拉开主物体、次要物体、背景之间的差距，释放出丰富的“灰色渐变”
        p_min, p_max = pred.min(), pred.max()
        if p_max - p_min > 1e-5:
            # 重新映射到完美的 0.0 ~ 1.0 区间
            pred = (pred - p_min) / (p_max - p_min)
        else:
            pred = np.zeros_like(pred)
            
        # 4. 转换成标准的 8 位灰度图
        pred = (pred * 255).astype(np.uint8)
        pred = cv2.resize(pred, (w, h))

        return pred