import os
import cv2
import numpy as np

TEST_DIR = "data/testA"
RESULT_DIR = "data/output/final_ranker"


def locate_crop_box(img_rgb, crop_rgb):
    """
    在原图中定位裁剪图的位置
    """

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)

    result = cv2.matchTemplate(
        img_gray,
        crop_gray,
        cv2.TM_CCOEFF_NORMED
    )

    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < 0.5:
        return None

    h, w = crop_gray.shape[:2]

    x1, y1 = max_loc

    return (
        x1,
        y1,
        x1 + w,
        y1 + h
    )


def compute_iou(box1, box2):

    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    ix1 = max(x11, x21)
    iy1 = max(y11, y21)

    ix2 = min(x12, x22)
    iy2 = min(y12, y22)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)

    inter = inter_w * inter_h

    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)

    union = area1 + area2 - inter

    if union <= 0:
        return 0.0

    return inter / union


def load_rgb(path):

    img = cv2.imread(path)

    if img is None:
        raise FileNotFoundError(path)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():

    results = []

    print("\n================ IoU Evaluation ================\n")

    for idx in range(1, 21):

        img_id = f"A{idx:02d}"

        original_path = os.path.join(
            TEST_DIR,
            f"{img_id}.jpg"
        )

        framing_path = os.path.join(
            TEST_DIR,
            f"{img_id}_framing.jpg"
        )

        pred_path = os.path.join(
            RESULT_DIR,
            f"{img_id}_best_crop.jpg"
        )

        try:

            original = load_rgb(original_path)
            gt_crop = load_rgb(framing_path)
            pred_crop = load_rgb(pred_path)

            gt_box = locate_crop_box(
                original,
                gt_crop
            )

            pred_box = locate_crop_box(
                original,
                pred_crop
            )

            if gt_box is None:
                print(f"{img_id} GT匹配失败")
                continue

            if pred_box is None:
                print(f"{img_id} Pred匹配失败")
                continue

            iou = compute_iou(
                pred_box,
                gt_box
            )

            results.append(iou)

            print(
                f"{img_id}   IoU = {iou:.4f}"
            )

        except Exception as e:

            print(
                f"{img_id} ERROR: {e}"
            )

    print("\n================================================\n")

    if len(results) > 0:

        mean_iou = np.mean(results)

        print(
            f"Mean IoU : {mean_iou:.4f}"
        )

        print(
            f"Images   : {len(results)}"
        )

        print(
            f"Max IoU  : {np.max(results):.4f}"
        )

        print(
            f"Min IoU  : {np.min(results):.4f}"
        )


if __name__ == "__main__":
    main()