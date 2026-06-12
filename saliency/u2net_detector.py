# saliency/u2net_detector.py

import torch
import cv2
import numpy as np

from saliency.models.u2net import U2NET


MODEL_PATH = r"D:\AI_Models\u2net.pth"


class U2NetDetector:

    _model = None

    def __init__(self):

        self.device = "cpu"

        if U2NetDetector._model is None:

            print("Loading U2Net...")

            net = U2NET(3, 1)

            net.load_state_dict(
                torch.load(
                    MODEL_PATH,
                    map_location=self.device
                )
            )

            net.eval()

            U2NetDetector._model = net

            print("U2Net loaded.")

        self.model = U2NetDetector._model

    def predict(self, image_rgb):

        h, w = image_rgb.shape[:2]

        img = cv2.resize(
            image_rgb,
            (320, 320)
        )

        img = img.astype(np.float32)

        img /= 255.0

        img = img.transpose(2, 0, 1)

        img = torch.tensor(
            img,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            d1, *_ = self.model(img)

        pred = d1[:, 0, :, :]

        pred = pred.squeeze()

        pred = pred.cpu().numpy()

        pred = (
            pred - pred.min()
        ) / (
            pred.max() - pred.min() + 1e-8
        )

        pred = (
            pred * 255
        ).astype(np.uint8)

        pred = cv2.resize(
            pred,
            (w, h)
        )

        return pred