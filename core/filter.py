# smart_framing/core/filter.py

import cv2
import numpy as np
from smart_framing.crop.bbox_utils import BBox, clip_bbox, compute_iou
from smart_framing.proposals.proposal_generator import generate_all_proposals
from smart_framing import config

# ---------- 工具函数：判断是否为风景mask ----------
def _is_landscape_mask(mask, landscape_masks, iou_thresh=0.3):
    total = mask.sum()
    if total == 0:
        return False
    for lm in landscape_masks:
        inter = np.logical_and(mask, lm).sum()
        union = np.logical_or(mask, lm).sum()
        if union > 0 and inter / union >= iou_thresh:
            return True
    return False

# ---------- build_instance_masks ----------
def build_instance_masks(img_rgb, saliency_mask, seg_map, segments,
                         depth_map=None,
                         saliency_thresh=0.5,
                         depth_near_ratio=0.35,
                         min_instance_area_ratio=0.002):
    h, w = img_rgb.shape[:2]
    img_area = h * w

    instance_masks = []
    landscape_masks = []
    sky_masks = []
    person_masks = []

    BACKGROUND_LABELS = {
        "wall", "ceiling", "ceiling-merged", "wall-other-merged",
        "floor", "floor-wood"
    }

    sal_binary = (saliency_mask > saliency_thresh).astype(np.uint8)
    num_sal_labels, sal_labels_im = cv2.connectedComponents(sal_binary)

    def get_inst_mask(seg):
        seg_id = seg.get("id", None)
        if seg_id is None:
            x1, y1, x2, y2 = seg["bbox"]
            m = np.zeros((h, w), dtype=np.uint8)
            m[y1:y2, x1:x2] = 1
            return m
        m_m2f = (seg_map == seg_id).astype(np.uint8)
        if m_m2f.sum() == 0:
            return m_m2f
        label_lower = seg["label"].lower()
        IS_BACKGROUND = any(bg in label_lower for bg in ["wall", "sky", "floor", "ceiling"])
        if IS_BACKGROUND:
            return m_m2f
        # 轨道 A：与 U2Net 联合
        matched_u2net = False
        m_final = m_m2f.copy()
        for label_id in range(1, num_sal_labels):
            m_sal = (sal_labels_im == label_id).astype(np.uint8)
            intersection = np.logical_and(m_m2f, m_sal).sum()
            if intersection > 0.3 * m_m2f.sum():
                m_final = np.logical_or(m_m2f, m_sal).astype(np.uint8)
                matched_u2net = True
                break
        # 轨道 B+C：深度召回
        if not matched_u2net:
            ys, xs = np.where(m_m2f > 0)
            if len(xs) > 0:
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                pad = 15
                bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
                bx2, by2 = min(w, x2 + pad), min(h, y2 + pad)
                m_fixed = m_m2f.copy()
                if depth_map is not None:
                    object_depths = depth_map[m_m2f > 0]
                    core_depth = np.median(object_depths)
                    depth_tolerance = 0.03
                    roi_depth = depth_map[by1:by2, bx1:bx2]
                    depth_match_roi = np.abs(roi_depth - core_depth) < depth_tolerance
                    depth_match_mask = np.zeros((h, w), dtype=np.uint8)
                    depth_match_mask[by1:by2, bx1:bx2] = depth_match_roi.astype(np.uint8)
                    dilated = cv2.dilate(m_fixed, np.ones((3, 3), np.uint8))
                    valid_depth_extensions = depth_match_mask & dilated
                    m_fixed = m_fixed | valid_depth_extensions
                # 搜寻小碎片
                roi_seg = seg_map[by1:by2, bx1:bx2]
                local_ids = np.unique(roi_seg)
                for loc_id in local_ids:
                    if loc_id == 0 or loc_id == seg_id:
                        continue
                    if np.sum(seg_map == loc_id) < 500:
                        m_fixed = m_fixed | (seg_map == loc_id).astype(np.uint8)
                restrict_mask = np.zeros((h, w), dtype=np.uint8)
                restrict_mask[by1:by2, bx1:bx2] = 1
                m_final = m_fixed & restrict_mask
        return m_final

    # 解析 segments
    for seg in segments:
        label_lower = seg["label"].lower()
        if label_lower in BACKGROUND_LABELS:
            continue
        is_person_like = label_lower in config.PERSON_LIKE_LABELS
        if not is_person_like:
            if seg.get("area", 0) < img_area * min_instance_area_ratio:
                continue
        mask = get_inst_mask(seg)
        if label_lower in config.PERSON_LIKE_LABELS:
            person_masks.append(mask)
            instance_masks.append(mask)
            continue
        is_sky = "sky" in label_lower
        is_landscape = any(kw in label_lower for kw in [
            "tree", "grass", "river", "water", "mountain", "road",
            "pavement", "plant", "sea", "lake", "wood", "building", "hill",
            "fence", "dirt", "field", "rock", "sand", "ground", "earth",
            "gravel", "playingfield", "terrain", "land", "banner", "floor"
        ])
        if is_sky:
            sky_masks.append(mask)
            landscape_masks.append(mask)
        elif is_landscape:
            landscape_masks.append(mask)
        else:
            instance_masks.append(mask)

    # U2Net 连通域
    sal_binary = (saliency_mask > saliency_thresh).astype(np.uint8)
    if sal_binary.sum() >= img_area * min_instance_area_ratio:
        num_labels, labels_im = cv2.connectedComponents(sal_binary)
        for label_id in range(1, num_labels):
            comp_mask = (labels_im == label_id).astype(np.uint8)
            if comp_mask.sum() >= img_area * min_instance_area_ratio:
                instance_masks.append(comp_mask)

    # Depth 近处连通域
    if depth_map is not None:
        depth_near = (depth_map > (1 - depth_near_ratio)).astype(np.uint8)
        if depth_near.sum() >= img_area * min_instance_area_ratio:
            num_labels, labels_im = cv2.connectedComponents(depth_near)
            for label_id in range(1, num_labels):
                comp_mask = (labels_im == label_id).astype(np.uint8)
                if comp_mask.sum() >= img_area * min_instance_area_ratio:
                    instance_masks.append(comp_mask)

    if len(instance_masks) == 0:
        instance_masks.append(np.ones((h, w), dtype=np.uint8))

    return instance_masks, landscape_masks, sky_masks, person_masks

# ---------- subject_coverage ----------
def subject_coverage(box, instance_masks):
    h, w = instance_masks[0].shape
    x1 = max(0, int(box.x1)); y1 = max(0, int(box.y1))
    x2 = min(w, int(box.x2)); y2 = min(h, int(box.y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0, -1
    best_cov = 0.0; best_idx = -1
    for idx, mask in enumerate(instance_masks):
        total = mask.sum()
        if total == 0: continue
        inside = mask[y1:y2, x1:x2].sum()
        cov = inside / total
        if cov > best_cov:
            best_cov = cov; best_idx = idx
    if best_idx == -1:
        return 0.0, -1
    return best_cov, best_idx

# ---------- NMS ----------
def nms_dedup(boxes, scores, iou_thresh=config.NMS_IOU_THRESH):
    order = np.argsort(scores)[::-1]
    keep = []
    suppressed = set()
    for idx in order:
        if idx in suppressed: continue
        keep.append(idx)
        for j in order:
            if j == idx or j in suppressed: continue
            if compute_iou(boxes[idx], boxes[j]) > iou_thresh:
                suppressed.add(j)
    return keep

# ---------- initial_filter ----------
def initial_filter(boxes, instance_masks, landscape_masks, sky_masks, person_masks,
                   img_w, img_h,
                   coverage_thresh=config.SUBJECT_COVERAGE_THRESH,
                   keep_top_n=config.KEEP_TOP_N):
    img_area = img_w * img_h
    img_cx, img_cy = img_w / 2, img_h / 2

    records = []
    for box in boxes:
        b = clip_bbox(box, img_w, img_h)
        if b.width <= 0 or b.height <= 0:
            continue
        cov, subj_idx = subject_coverage(b, instance_masks)

        # 截断惩罚 (原有)
        clip_penalty = 0.0
        if subj_idx is not None and subj_idx >= 0 and subj_idx < len(instance_masks):
            target_mask = instance_masks[subj_idx]
            total_subject_pixels = target_mask.sum()
            if total_subject_pixels > 0:
                inter_pixels = target_mask[int(b.y1):int(b.y2), int(b.x1):int(b.x2)].sum()
                clipped_ratio = 1.0 - cov
                if inter_pixels > 0 and clipped_ratio > 0.05:
                    clip_penalty = 0.6 * (clipped_ratio / 0.95)

        # 人物独立截断惩罚
        person_clip_penalty = 0.0
        for p_mask in person_masks:
            total = p_mask.sum()
            if total == 0: continue
            inter = p_mask[int(b.y1):int(b.y2), int(b.x1):int(b.x2)].sum()
            if inter == 0: continue
            p_cov = inter / total
            if p_cov < 0.98:
                clipped = 1.0 - p_cov
                this_penalty = 0.8 * min(1.0, clipped / 0.3)
                person_clip_penalty = max(person_clip_penalty, this_penalty)

        # 风景覆盖率
        land_cov = 0.0
        if len(landscape_masks) > 0:
            land_cov, _ = subject_coverage(b, landscape_masks)

        # 天空占比
        sky_ratio_in_box = 0.0
        if len(sky_masks) > 0:
            sky_inter_pixels = sum([mask[int(b.y1):int(b.y2), int(b.x1):int(b.x2)].sum() for mask in sky_masks])
            sky_ratio_in_box = sky_inter_pixels / b.area if b.area > 0 else 0.0

        area_ratio = b.area / img_area
        if area_ratio < 0.12:
            area_score = area_ratio / 0.12
        elif area_ratio > 0.65:
            area_score = max(0.4, 1.0 - 0.4 * ((area_ratio - 0.65) / 0.35))
        else:
            area_score = 1.0

        dx = abs(b.center_x - img_cx) / (img_w / 2)
        dy = abs(b.center_y - img_cy) / (img_h / 2)
        center_score = max(0.0, 1.0 - 0.4 * (dx + dy))

        is_landscape_intent = (land_cov >= 0.25) and (land_cov > cov * 0.8)

        if is_landscape_intent:
            composition_score = 0.4 * cov + 0.6 * land_cov
        else:
            composition_score = cov

        score = 0.5 * composition_score + 0.3 * area_score + 0.2 * center_score

        if sky_ratio_in_box >= 0.70:
            penalty = 0.5 * ((sky_ratio_in_box - 0.70) / 0.30)
            score = max(0.0, score - penalty)

        score = max(0.0, score - clip_penalty)
        score = max(0.0, score - person_clip_penalty)

        records.append({
            "box": b,
            "coverage": cov,
            "land_coverage": land_cov,
            "subject_idx": subj_idx,
            "area_ratio": area_ratio,
            "is_landscape": is_landscape_intent,
            "clip_penalty": clip_penalty,
            "person_clip_penalty": person_clip_penalty,
            "score": score,
        })

    # 硬过滤
    filtered = [
        r for r in records
        if r["person_clip_penalty"] < 0.2
        and ((r["coverage"] >= coverage_thresh and r["clip_penalty"] < 0.2) or
             (r["is_landscape"] and r["land_coverage"] >= 0.3 and r["clip_penalty"] < 0.2))
    ]
    if len(filtered) < keep_top_n:
        filtered = sorted(records, key=lambda x: x["score"], reverse=True)[:keep_top_n]

    # 动态配额
    if len(landscape_masks) > 0:
        landscape_union = np.zeros_like(landscape_masks[0])
        for m in landscape_masks:
            landscape_union = np.logical_or(landscape_union, m)
        total_landscape_ratio = landscape_union.sum() / img_area
    else:
        total_landscape_ratio = 0.0
    total_landscape_ratio = min(1.0, total_landscape_ratio)
    land_quota_ratio = 0.2 + 0.3 * total_landscape_ratio
    land_quota = int(round(keep_top_n * land_quota_ratio))
    sub_quota = keep_top_n - land_quota

    subject_channel = []
    landscape_channel = []
    for r in filtered:
        if r["is_landscape"]:
            landscape_channel.append(r)
        else:
            subject_channel.append(r)

    def fill_to_quota(records_in, pool):
        if len(records_in) >= keep_top_n:
            return records_in[:keep_top_n] if len(records_in) > keep_top_n else records_in
        def box_key(r):
            b = r["box"]
            return (round(b.x1, 1), round(b.y1, 1), round(b.x2, 1), round(b.y2, 1))
        existing_keys = {box_key(r) for r in records_in}
        all_sorted = sorted(pool, key=lambda r: r["score"], reverse=True)
        out = list(records_in)
        for r in all_sorted:
            if len(out) >= keep_top_n:
                break
            k = box_key(r)
            if k not in existing_keys:
                out.append(r); existing_keys.add(k)
        out.sort(key=lambda r: r["score"], reverse=True)
        return out

    if len(landscape_channel) == 0 or len(subject_channel) == 0:
        boxes_f = [r["box"] for r in filtered]
        scores_f = [r["score"] for r in filtered]
        keep_idx = nms_dedup(boxes_f, scores_f, config.NMS_IOU_THRESH)
        kept = [filtered[i] for i in keep_idx]
        kept.sort(key=lambda r: r["score"], reverse=True)
        kept = kept[:keep_top_n]
        kept = fill_to_quota(kept, records)
        return kept, records

    sub_boxes = [r["box"] for r in subject_channel]
    sub_scores = [r["score"] for r in subject_channel]
    sub_keep = nms_dedup(sub_boxes, sub_scores, config.NMS_IOU_THRESH)
    sub_kept = [subject_channel[i] for i in sub_keep]
    sub_kept.sort(key=lambda r: r["score"], reverse=True)

    land_boxes = [r["box"] for r in landscape_channel]
    land_scores = [r["score"] for r in landscape_channel]
    land_keep = nms_dedup(land_boxes, land_scores, iou_thresh=0.88)
    land_kept = [landscape_channel[i] for i in land_keep]
    land_kept.sort(key=lambda r: r["score"], reverse=True)

    final_land = land_kept[:land_quota]
    final_sub = sub_kept[:sub_quota]

    if len(final_land) < land_quota:
        extra = land_quota - len(final_land)
        final_sub = sub_kept[:sub_quota + extra]
    elif len(final_sub) < sub_quota:
        extra = sub_quota - len(final_sub)
        final_land = land_kept[:land_quota + extra]

    final_records = final_sub + final_land
    final_records.sort(key=lambda r: r["score"], reverse=True)
    final_records = fill_to_quota(final_records, records)
    return final_records, records