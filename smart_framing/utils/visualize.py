# utils/visualize.py
import cv2
import numpy as np
from smart_framing import config

def draw_overlay(img_rgb, records, framing_box=None):
    """绘制半透明叠加框（原 filter_and_visualize.draw_overlay）"""
    img = img_rgb.copy()
    overlay = img.copy()
    scores = np.array([r["score"] for r in records])
    s_min, s_max = scores.min(), scores.max()
    norm = (scores - s_min) / (s_max - s_min + 1e-6)
    for r, n in zip(records, norm):
        b = r["box"]
        color = (int(255 * (1 - n)), int(255 * n), 0)
        cv2.rectangle(overlay, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 1)
    blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
    if framing_box is not None:
        cv2.rectangle(blended,
                      (int(framing_box.x1), int(framing_box.y1)),
                      (int(framing_box.x2), int(framing_box.y2)),
                      (255, 255, 255), 3)
    return blended

def draw_final_top_k(img_rgb, records, k=20, framing_img=None):
    """
    等比例缩放并拼接 TopK 裁剪（原 final_ranker.draw_final_top_k）
    """
    TARGET_W, TARGET_H = 160, 160
    crops = []
    for r in records[:k]:
        b = r["box"]
        x1, y1 = int(b.x1), int(b.y1)
        x2, y2 = int(b.x2), int(b.y2)
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        h_crop, w_crop = crop.shape[:2]
        scale = min(TARGET_W / w_crop, TARGET_H / h_crop)
        new_w = int(w_crop * scale)
        new_h = int(h_crop * scale)
        crop_resized = cv2.resize(crop, (new_w, new_h))
        canvas = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
        dx = (TARGET_W - new_w) // 2
        dy = (TARGET_H - new_h) // 2
        canvas[dy:dy+new_h, dx:dx+new_w] = crop_resized

        label1 = f"fin={r['final_score']:.2f} aes={r['aes_norm']:.2f}"
        label2 = f"con={r['content_score']:.2f} 3rd={r['thirds_score']:.2f}"
        msg_penalty = config.W_MISSING_PENALTY if r.get("missing_subject", 0.0) > 0 else 0.0
        label3 = f"ctr={r['center_score']:.2f} msg=-{msg_penalty:.1f} pen={r['object_clip_penalty']:.1f}"
        canvas = cv2.copyMakeBorder(canvas, 48, 0, 0, 0,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(canvas, label1, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255,255,255), 1)
        cv2.putText(canvas, label2, (2, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255,255,255), 1)
        cv2.putText(canvas, label3, (2, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255,255,255), 1)
        crops.append(canvas)

    if framing_img is not None:
        fh, fw = framing_img.shape[:2]
        f_scale = min(TARGET_W / fw, TARGET_H / fh)
        fn_w, fn_h = int(fw * f_scale), int(fh * f_scale)
        f_resized = cv2.resize(framing_img, (fn_w, fn_h))
        f_canvas = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
        fdx = (TARGET_W - fn_w) // 2
        fdy = (TARGET_H - fn_h) // 2
        f_canvas[fdy:fdy+fn_h, fdx:fdx+fn_w] = f_resized
        f_canvas = cv2.copyMakeBorder(f_canvas, 48, 0, 0, 0,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 255))
        cv2.putText(f_canvas, "GT framing", (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        crops.insert(0, f_canvas)

    cols = 5
    rows = (len(crops) + cols - 1) // cols
    grid = np.zeros((rows * 208, cols * 160, 3), dtype=np.uint8)
    for i, c in enumerate(crops):
        r_idx, col_idx = divmod(i, cols)
        grid[r_idx * 208: r_idx * 208 + 208, col_idx * 160: col_idx * 160 + 160] = c
    return grid

# 可选：draw_top_k 用于旧版网格，可保留但此处不再重复