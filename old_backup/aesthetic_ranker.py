import cv2
import torch
import torch.nn as nn
import open_clip

from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading CLIP...")

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14",
    pretrained="openai"
)

clip_model = clip_model.to(DEVICE)
clip_model.eval()

class AestheticPredictor(nn.Module):

    def __init__(self):

        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
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
    
    
MODEL_PATH = r"weights/ava+logos-l14-linearMSE.pth"

predictor = AestheticPredictor()

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

predictor.load_state_dict(state_dict)

predictor = predictor.to(DEVICE)
predictor.eval()

print("Aesthetic predictor loaded.")

def extract_feature(rgb_img):

    image = Image.fromarray(rgb_img)

    image_tensor = preprocess(image)

    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        feature = clip_model.encode_image(
            image_tensor
        )

        feature /= feature.norm(
            dim=-1,
            keepdim=True
        )

    return feature

def aesthetic_score(rgb_img):

    feature = extract_feature(rgb_img)

    with torch.no_grad():

        score = predictor(feature)

    return float(score.cpu().item())

def aesthetic_rerank(
        img_rgb,
        final_records
):

    print("\n开始美学评分...")

    for i, record in enumerate(final_records):

        box = record["box"]

        crop = img_rgb[
            box.y1:box.y2,
            box.x1:box.x2
        ]

        if crop.size == 0:

            record["aesthetic_score"] = -999

            continue

        score = aesthetic_score(crop)

        record["aesthetic_score"] = score

        if i % 10 == 0:
            print(
                f"{i}/{len(final_records)}"
            )

    final_records.sort(
        key=lambda x: x["aesthetic_score"],
        reverse=True
    )

    return final_records

