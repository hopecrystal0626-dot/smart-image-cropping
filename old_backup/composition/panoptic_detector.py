import torch
import numpy as np
from PIL import Image

from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation
)


class PanopticDetector:

    def __init__(
        self,
        model_name="facebook/mask2former-swin-base-coco-panoptic"
    ):
        print("Loading Mask2Former...")

        self.device = "cpu"

        self.processor = AutoImageProcessor.from_pretrained(
            model_name
        )

        self.model = (
            Mask2FormerForUniversalSegmentation
            .from_pretrained(model_name)
        )

        self.model.to(self.device)
        self.model.eval()

        print("Mask2Former loaded.")

    def predict(self, image_rgb):
        """
        输入:
            image_rgb (numpy RGB)

        返回:
            result:
            {
                "segmentation": tensor(H,W),
                "segments_info": [...]
            }
        """

        pil_img = Image.fromarray(image_rgb)

        inputs = self.processor(
            images=pil_img,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        result = (
            self.processor
            .post_process_panoptic_segmentation(
                outputs,
                target_sizes=[
                    (image_rgb.shape[0],
                     image_rgb.shape[1])
                ]
            )[0]
        )

        return result

    def build_scene_vector(self, result):
        """
        根据panoptic结果统计面积占比

        返回:
        {
            "building":0.42,
            "sky":0.31,
            ...
        }
        """

        seg_map = result["segmentation"].cpu().numpy()

        h, w = seg_map.shape
        total_pixels = h * w

        scene_vector = {}

        for seg in result["segments_info"]:

            segment_id = seg["id"]

            label_id = seg["label_id"]

            label_name = (
                self.model.config.id2label
                .get(label_id, str(label_id))
            )

            area = np.sum(seg_map == segment_id)

            ratio = area / total_pixels

            scene_vector[label_name] = (
                scene_vector.get(label_name, 0)
                + ratio
            )

        return scene_vector
    
    def get_segment_bboxes(self, result):

        seg_map = result["segmentation"].cpu().numpy()

        segments = []

        for seg in result["segments_info"]:

            segment_id = seg["id"]

            label_id = seg["label_id"]

            label_name = self.model.config.id2label.get(
                label_id,
                str(label_id)
            )

            ys, xs = np.where(seg_map == segment_id)

            if len(xs) == 0:
                continue

            x1 = xs.min()
            x2 = xs.max()

            y1 = ys.min()
            y2 = ys.max()

            area = len(xs)

            segments.append({
                "label": label_name,
                "bbox": (x1, y1, x2, y2),
                "area": area,
                "score": seg["score"]
            })

        return segments