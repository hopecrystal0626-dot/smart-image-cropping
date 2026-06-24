# models/u2net_detector.py
import torch
import cv2
import numpy as np
from smart_framing.models.u2net import U2NET
from collections import OrderedDict

class U2NetDetector:
    _model = None
    def __init__(self, model_path="weights/u2net.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if U2NetDetector._model is None:
            net = U2NET(3, 1)
            state_dict = torch.load(model_path, map_location=self.device)
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

        self.model = U2NetDetector._model

    def predict(self, image_rgb):
        h, w = image_rgb.shape[:2]
        
        img = cv2.resize(image_rgb, (320, 320))
        img = img.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        img = img.transpose(2, 0, 1)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            d0, d1, d2, d3, d4, d5, d6 = self.model(img)
            # 优化细节：改用 d0 作为输出，它是官方标准的最佳融合图
            mix_tensor = d0 
            pred_tensor = mix_tensor[0, 0, :, :].squeeze()

        pred = pred_tensor.cpu().numpy()

        p_min, p_max = pred.min(), pred.max()
        if p_max - p_min > 1e-5:
            pred = (pred - p_min) / (p_max - p_min)
        else:
            pred = np.zeros_like(pred)
            
        pred = (pred * 255).astype(np.uint8)
        pred = cv2.resize(pred, (w, h))

        return pred