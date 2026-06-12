import torch
import numpy as np
from PIL import Image

from transformers import AutoModelForImageSegmentation


class ISNetDetector:

    def __init__(
        self,
        model_name="briaai/RMBG-2.0"
    ):
        print("Loading IS-Net (RMBG-2.0)...")

        self.device = "cpu"

        self.model = (
            AutoModelForImageSegmentation
            .from_pretrained(
                model_name,
                trust_remote_code=True
            )
        )

        self.model.to(self.device)
        self.model.eval()

        print("IS-Net loaded.")

    def predict(self, image_rgb):

        image = (
            Image
            .fromarray(image_rgb)
            .convert("RGB")
        )

        image = image.resize((1024, 1024))

        image_np = np.array(image)

        image_tensor = (
            torch.tensor(image_np)
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():

            pred = self.model(
                image_tensor
            )[-1]

            pred = (
                torch.sigmoid(pred)
                .cpu()
                .numpy()[0, 0]
            )

        pred = (
            pred * 255
        ).astype(np.uint8)

        return pred