# core/pipeline.py
import os
import cv2
import numpy as np
from smart_framing.core.inference import compute_saliency, compute_panoptic, compute_depth
from smart_framing.proposals.proposal_generator import generate_all_proposals
from smart_framing.core.filter import build_instance_masks, initial_filter
from smart_framing.core.ranker import fuse_and_rank
from smart_framing.utils.helpers import load_image, load_framing
from smart_framing.utils.visualize import draw_final_top_k
from smart_framing import config

def process_image(img_path, use_depth=True, save_vis=True, output_dir=None):
    """
    处理单张图片，返回最佳框、Top10、可视化等。
    Args:
        img_path: 任意图片路径（绝对或相对）
        use_depth: 是否启用深度模型
        save_vis: 是否保存可视化图片
        output_dir: 保存目录，默认 config.OUTPUT_DIR
    Returns:
        dict: {
            'best_box': (x1,y1,x2,y2),
            'best_crop': np.ndarray (RGB),
            'top10': list of dict,
            'grid_image': np.ndarray (可选),
            'all_records': list
        }
    """
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    img_rgb = load_image(img_path)
    h, w = img_rgb.shape[:2]
    framing_img = load_framing(img_path)   # 若存在同名 _framing.jpg 则加载，否则 None

    # 1. 模型推理
    saliency_mask = compute_saliency(img_rgb)                     # 0~1 float
    pan_result, segments, seg_map = compute_panoptic(img_rgb)
    depth_map = compute_depth(img_rgb) if use_depth else None

    # 2. 候选框生成（saliency 需转为 uint8 0~255）
    saliency_uint8 = (saliency_mask * 255).astype(np.uint8)
    all_boxes = generate_all_proposals(img_rgb, saliency_uint8, segments)

    # 3. 构建实例 masks
    instance_masks, landscape_masks, sky_masks, person_masks = build_instance_masks(
        img_rgb, saliency_mask, seg_map, segments, depth_map
    )

    # 4. 初筛
    final_records, _ = initial_filter(
        all_boxes, instance_masks, landscape_masks, sky_masks, person_masks, w, h
    )

    # 5. 融合排序
    ranked = fuse_and_rank(img_rgb, final_records, instance_masks, landscape_masks, person_masks, depth_map)

    # 6. 提取结果
    best = ranked[0]
    best_box = (int(best['box'].x1), int(best['box'].y1), int(best['box'].x2), int(best['box'].y2))
    best_crop = img_rgb[best_box[1]:best_box[3], best_box[0]:best_box[2]]

    top10 = [{
        'box': (int(r['box'].x1), int(r['box'].y1), int(r['box'].x2), int(r['box'].y2)),
        'score': r['final_score'],
        'aes': r['aes_norm'],
        'content': r['content_score'],
        'thirds': r['thirds_score'],
        'center': r['center_score']
    } for r in ranked[:10]]
    
    top10_crops = []
    for r in ranked[:10]:
        b = r["box"]
        x1 = max(0, int(b.x1))
        y1 = max(0, int(b.y1))
        x2 = min(w, int(b.x2))
        y2 = min(h, int(b.y2))
        crop = img_rgb[y1:y2, x1:x2]
        top10_crops.append(crop)  # 直接存 numpy 数组

    # 7. 可视化（可选）
    grid_img = None
    if save_vis:
        out_dir = output_dir or config.OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)
        grid_img = draw_final_top_k(img_rgb, ranked, k=20, framing_img=framing_img)
        cv2.imwrite(os.path.join(out_dir, f"{img_id}_final_grid.jpg"), cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out_dir, f"{img_id}_best_crop.jpg"), cv2.cvtColor(best_crop, cv2.COLOR_RGB2BGR))

    return {
        'best_box': best_box,
        'best_crop': best_crop,
        'top10': top10,
        'top10_crops': top10_crops,
        'grid_image': grid_img,
        'all_records': ranked
    }