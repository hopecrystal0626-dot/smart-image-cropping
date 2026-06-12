from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import List, Sequence, Tuple

from composition.aesthetic_scorer import AestheticScorer
from composition.composition_score import compute_composition_score_single
from composition.human_detector import HumanDetector
from composition.object_detector import ObjectDetector
from crop.bbox_utils import BBox
from crop.candidate_generator import generate_candidates
from clip_score import get_scorer
from saliency.detector import FTDetector
from saliency.saliency_utils import filter_top_bboxes_by_percentile

from .config import DEFAULT_PIPELINE_CONFIG, PipelineConfig


class SystemizedSmartCropping:
    def __init__(self, config: PipelineConfig = DEFAULT_PIPELINE_CONFIG):
        self.config = config
        print("1")
        self.aesthetic_scorer = AestheticScorer()
        print("2")
        self.clip_scorer = get_scorer(mode=config.fusion.clip_mode)
        print("3")
        self.saliency_detector = FTDetector()
        print("4")
        self.human_detector = HumanDetector()
        print("5")
        self.object_detector = ObjectDetector(conf_threshold=config.completeness.yolo_conf_threshold)
        print("6")

    def generate_candidates(self, image_w: int, image_h: int):
        return generate_candidates(
            image_w,
            image_h,
            jitter_ratio=self.config.candidate.jitter_ratio,
            seed=self.config.candidate.seed,
            scale_grid=self.config.candidate.scale_grid,
        )

    def filter_by_saliency(self, image: np.ndarray, candidates):
        sal_map = self.saliency_detector.detect(image)
        return filter_top_bboxes_by_percentile(
            sal_map,
            candidates,
            top_percent=self.config.saliency.top_percent,
            num_segments=self.config.saliency.num_segments,
        ), sal_map

    def detect_subjects(self, image: np.ndarray):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        human_boxes, object_boxes = self.object_detector.detect_all(img_rgb, verbose=False)
        if not human_boxes:
            human_boxes = self.human_detector.detect_human_bboxes(img_rgb)
        return human_boxes, object_boxes

    def _bbox_tuple_to_bbox(self, box: Tuple[int, int, int, int]) -> BBox:
        x1, y1, x2, y2 = box
        return BBox(int(x1), int(y1), int(x2), int(y2), scale=1.0)

    def _clamp_bbox(self, x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> BBox:
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(x1 + 1, min(int(x2), w))
        y2 = max(y1 + 1, min(int(y2), h))
        return BBox(x1, y1, x2, y2, scale=1.0)

    def _max_coverage(self, candidate_bbox: BBox, boxes: Sequence[Tuple[int, int, int, int]]) -> float:
        if not boxes:
            return 0.0
        cx1, cy1, cx2, cy2 = candidate_bbox.x1, candidate_bbox.y1, candidate_bbox.x2, candidate_bbox.y2
        best = 0.0
        for ox1, oy1, ox2, oy2 in boxes:
            ix1 = max(cx1, ox1)
            iy1 = max(cy1, oy1)
            ix2 = min(cx2, ox2)
            iy2 = min(cy2, oy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter_area = (ix2 - ix1) * (iy2 - iy1)
            o_area = max(1, (ox2 - ox1) * (oy2 - oy1))
            best = max(best, inter_area / o_area)
        return best

    def _edge_touch_ratio(self, candidate_bbox: BBox, boxes: Sequence[Tuple[int, int, int, int]]) -> float:
        if not boxes:
            return 0.0
        margin = self.config.completeness.edge_margin_px
        touch_count = 0
        for ox1, oy1, ox2, oy2 in boxes:
            if abs(candidate_bbox.x1 - ox1) <= margin:
                touch_count += 1
            elif abs(candidate_bbox.y1 - oy1) <= margin:
                touch_count += 1
            elif abs(candidate_bbox.x2 - ox2) <= margin:
                touch_count += 1
            elif abs(candidate_bbox.y2 - oy2) <= margin:
                touch_count += 1
        return touch_count / max(1, len(boxes))

    def _scene_statistics(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        vertical_energy = float(np.mean(np.abs(gx)))
        horizontal_energy = float(np.mean(np.abs(gy)))
        total_energy = vertical_energy + horizontal_energy + 1e-6
        vertical_ratio = vertical_energy / total_energy
        horizontal_ratio = horizontal_energy / total_energy
        edge_density = float(np.mean(edges > 0))

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=max(20, min(image.shape[:2]) // 8), maxLineGap=15)
        line_count = 0 if lines is None else len(lines)
        line_total = 0.0
        vertical_lines = 0.0
        horizontal_lines = 0.0
        if lines is not None:
            for line in lines[:, 0]:
                x1, y1, x2, y2 = line.tolist()
                length = float(np.hypot(x2 - x1, y2 - y1))
                if length < 12:
                    continue
                line_total += length
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                angle = min(angle, 180 - angle)
                if angle <= 20:
                    horizontal_lines += length
                elif angle >= 70:
                    vertical_lines += length

        line_total = max(line_total, 1e-6)
        line_balance = (vertical_lines + horizontal_lines) / line_total

        return {
            "vertical_ratio": vertical_ratio,
            "horizontal_ratio": horizontal_ratio,
            "edge_density": edge_density,
            "line_count": line_count,
            "line_balance": line_balance,
        }

    def _classify_scene(self, image: np.ndarray, human_boxes, object_boxes):
        print("CLASSIFY FUNCTION ENTER")
        print("=" * 60)
        print("human_boxes =", len(human_boxes))
        print("object_boxes =", len(object_boxes))
        print("=" * 60)
        
        stats = self._scene_statistics(image)
        

        if len(human_boxes) > 0:
            print("[Scene] portrait")
            return "portrait", stats

        

        building_score = 0.0
        if stats["vertical_ratio"] >= 0.52:
            building_score += 0.30
        if stats["horizontal_ratio"] >= 0.48:
            building_score += 0.20
        if stats["edge_density"] >= 0.04:
            building_score += 0.20
        if stats["line_balance"] >= 0.35:
            building_score += 0.30
            
        subject_score = 0.0
        if len(object_boxes) > 0:
            subject_score += 0.6
            img_area = image.shape[0] * image.shape[1]
            max_ratio = 0.0
            for x1, y1, x2, y2 in object_boxes:
                area = (x2 - x1) * (y2 - y1)
                ratio = area / img_area
                
                if ratio > max_ratio:
                    max_ratio = ratio
                    
            if max_ratio > 0.05:
                subject_score += 0.4
            if max_ratio > 0.10:
                subject_score += 0.2
            if max_ratio > 0.20:
                subject_score += 0.20
        
            large_count = sum(
                1 for x1, y1, x2, y2 in object_boxes
                if (x2 - x1) * (y2 - y1) / img_area > 0.05
            )
            
            if large_count >= 2:
                subject_score += 0.1
                

        if subject_score >= 0.60:
            scene_type = "subject"
            
        elif building_score >= 0.70:
            scene_type = "building"
            
        else:
            scene_type = "landscape"
        

        print(
            f"[Scene] {scene_type} | "
            f"building_score={building_score:.3f} "
            f"subject={subject_score:.2f} "
            f"objects={len(object_boxes)} "
            f"vertical={stats['vertical_ratio']:.3f} "
            f"horizontal={stats['horizontal_ratio']:.3f} "
            f"edge={stats['edge_density']:.3f} "
            f"line={stats['line_balance']:.3f}"
        )
        return scene_type, stats

    def _structure_candidates(self, image: np.ndarray, sal_map: np.ndarray):
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        if sal_map is not None:
            sal_binary = (sal_map > np.percentile(sal_map, 75)).astype(np.uint8)
            edges = cv2.bitwise_or(edges, (sal_binary * 255).astype(np.uint8))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        dilated = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((dilated > 0).astype(np.uint8), 8)
        boxes = []
        min_area = max(0.01, self.config.completeness.min_area_ratio * 0.35) * (w * h)
        pad_ratio = self.config.scene.structure_pad_ratio

        for idx in range(1, num_labels):
            x, y, bw, bh, area = stats[idx]
            if area < min_area:
                continue
            if bw < 20 or bh < 20:
                continue
            x1 = x - int(bw * pad_ratio)
            y1 = y - int(bh * pad_ratio)
            x2 = x + bw + int(bw * pad_ratio)
            y2 = y + bh + int(bh * pad_ratio)
            boxes.append(self._clamp_bbox(x1, y1, x2, y2, w, h))

        # 兜底：基于行/列投影生成少量结构候选
        col_energy = np.sum(edges > 0, axis=0)
        row_energy = np.sum(edges > 0, axis=1)
        if col_energy.max() > 0 and row_energy.max() > 0:
            x_peak = int(np.argmax(col_energy))
            y_peak = int(np.argmax(row_energy))
            box_w = max(int(w * 0.55), 60)
            box_h = max(int(h * 0.55), 60)
            boxes.append(self._clamp_bbox(x_peak - box_w // 2, y_peak - box_h // 2, x_peak + box_w // 2, y_peak + box_h // 2, w, h))

        # 去重并限制数量
        unique = []
        seen = set()
        for b in boxes:
            key = (b.x1, b.y1, b.x2, b.y2)
            if key in seen:
                continue
            seen.add(key)
            unique.append(b)
        unique.sort(key=lambda b: b.area, reverse=True)
        return unique[: self.config.scene.structure_candidate_max]

    def _structure_score(self, image: np.ndarray, bbox: BBox) -> float:
        crop = image[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        vertical_energy = float(np.mean(np.abs(gx)))
        horizontal_energy = float(np.mean(np.abs(gy)))
        total_energy = vertical_energy + horizontal_energy + 1e-6
        line_balance = max(vertical_energy, horizontal_energy) / total_energy
        edge_density = float(np.mean(edges > 0))

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=max(10, min(crop.shape[:2]) // 10), maxLineGap=10)
        aligned = 0.0
        total = 0.0
        if lines is not None:
            for line in lines[:, 0]:
                x1, y1, x2, y2 = line.tolist()
                length = float(np.hypot(x2 - x1, y2 - y1))
                if length < 8:
                    continue
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                angle = min(angle, 180 - angle)
                total += length
                if angle <= 18 or angle >= 72:
                    aligned += length
        alignment_ratio = aligned / max(total, 1e-6)

        center_x = (bbox.x1 + bbox.x2) / 2 / image.shape[1]
        center_y = (bbox.y1 + bbox.y2) / 2 / image.shape[0]
        center_bias = 1.0 - min(1.0, abs(center_x - 0.5) * 1.1 + abs(center_y - 0.5) * 1.1)

        score = 0.42 * alignment_ratio + 0.28 * edge_density * 4.0 + 0.20 * line_balance + 0.10 * center_bias
        return float(np.clip(score, 0.0, 1.0))

    def save_best_crop(self, image_path: str, best_result: dict, output_dir: str):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        bbox = best_result["bbox"]
        crop = img[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        if crop.size == 0:
            raise ValueError("最佳候选框裁剪结果为空")

        import os
        from pathlib import Path

        os.makedirs(output_dir, exist_ok=True)
        base = Path(image_path).stem
        crop_path = Path(output_dir) / f"{base}_best_crop.jpg"
        vis_path = Path(output_dir) / f"{base}_best_bbox.jpg"

        cv2.imwrite(str(crop_path), crop)

        canvas = img.copy()
        cv2.rectangle(canvas, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (255, 0, 255), 3)
        cv2.putText(canvas, f"best={best_result['rerank_score']:.3f}", (bbox.x1, max(20, bbox.y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.imwrite(str(vis_path), canvas)

        return str(crop_path), str(vis_path)

    def save_topk_visualization(self, image_path: str, topk_results: list, output_dir: str, k: int = 5):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")

        canvas = img.copy()
        colors = [
            (255, 0, 255),
            (0, 255, 0),
            (0, 255, 255),
            (255, 165, 0),
            (255, 0, 0),
        ]

        for i, item in enumerate(topk_results[:k]):
            bbox = item["bbox"]
            color = colors[i % len(colors)]
            thickness = 3 if i == 0 else 2
            cv2.rectangle(canvas, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, thickness)
            cv2.putText(
                canvas,
                f"#{i+1} {item['rerank_score']:.3f}",
                (bbox.x1, max(20, bbox.y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        import os

        os.makedirs(output_dir, exist_ok=True)
        save_path = Path(output_dir) / f"{Path(image_path).stem}_top{k}.jpg"
        cv2.imwrite(str(save_path), canvas)
        return str(save_path)

    def score_with_fusion(self, image: np.ndarray, candidates, scene_type: str = "portrait"):
        h, w = image.shape[:2]
        scored_boxes = []
        for bbox in candidates:
            cropped = image[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
            if cropped.size == 0:
                continue
            detail = compute_composition_score_single(
                cropped,
                bbox=(bbox.x1, bbox.y1, bbox.width, bbox.height),
            )
            handcraft_score = detail["total_score"]
            try:
                nima_score = self.aesthetic_scorer.get_score_for_bbox(image, bbox) / 10.0
            except Exception:
                nima_score = 0.5

            structure_score = 0.0

            if scene_type == "portrait":
                fusion_score = (
                    self.config.fusion.alpha * handcraft_score
                    + self.config.fusion.beta * nima_score
                )

            elif scene_type == "subject":
            # 主体场景：构图+美学为主，加入少量结构感知
                structure_score = self._structure_score(image, bbox)
                fusion_score = (
                    0.40 * handcraft_score
                    + 0.55 * nima_score
                    + 0.05 * structure_score
                )

            elif scene_type == "building":
            # 建筑场景：结构对齐最重要，构图其次
                structure_score = self._structure_score(image, bbox)
                fusion_score = (
                    0.25 * handcraft_score
                    + 0.35 * nima_score
                    + 0.40 * structure_score
                )

            else:  # landscape
            # 风景：美学+构图均衡，结构感知辅助
                structure_score = self._structure_score(image, bbox)
                fusion_score = (
                    0.45 * handcraft_score
                    + 0.50 * nima_score
                    + 0.05 * structure_score
                )

            scored_boxes.append({
                "bbox": bbox,
                "handcraft_score": handcraft_score,
                "nima_score": nima_score,
                "structure_score": structure_score,
                "fusion_score": fusion_score,
                "area_ratio": (bbox.width * bbox.height) / (w * h),
            })
        scored_boxes.sort(key=lambda x: x["fusion_score"], reverse=True)
        return scored_boxes


    def run(self, image_path: str, top_k: int | None = None):
        top_k = top_k or self.config.fusion.top_k
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")

        h, w = img.shape[:2]
        candidates = [
            bbox for bbox in self.generate_candidates(w, h)
            if (bbox.width * bbox.height) / (w * h) >= self.config.completeness.min_area_ratio
        ]
   
        human_boxes, object_boxes = self.detect_subjects(img)
        scene_type, scene_stats = self._classify_scene(img, human_boxes, object_boxes)
        print("=" * 60)
        print("scene_type =", scene_type)
        print("human_boxes =", len(human_boxes))
        print("object_boxes =", len(object_boxes))
        print("=" * 60)
        
        (top_bboxes, _, _), sal_map = self.filter_by_saliency(
            img,
            candidates
        )
        
        if scene_type in ("portrait", "subject"):
            final_candidates = list(top_bboxes)
            print(f"[Candidates] {scene_type}: using saliency top_bboxes only, count={len(final_candidates)}")
            
        elif scene_type == "building":
            sal_set = {(b.x1, b.y1, b.x2, b.y2): b for b in top_bboxes}
            structure_candidates = self._structure_candidates(img, sal_map)
            struct_set = {(b.x1, b.y1, b.x2, b.y2): b for b in structure_candidates}
            
            merged = list(top_bboxes)
            seen = {(b.x1, b.y1, b.x2, b.y2) for b in merged}
            
            structure_supplement_max = self.config.scene.structure_candidate_max // 2
            added = 0
            for bbox in structure_candidates:
                key = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
                if key in seen:
                    continue
                has_overlap = any(
                    self._iou(bbox, sal_bbox) > 0.10
                    for sal_bbox in top_bboxes
                )
                if not has_overlap and len(sal_set) > 0:
                    continue
                seen.add(key)
                merged.append(bbox)
                added += 1
                if added >= structure_supplement_max:
                    break
                
            final_candidates = merged
            print(f"[Candidates] building: saliency={len(top_bboxes)}, struct_added={added}, total={len(final_candidates)}")
            
        else:
            final_candidates = list(top_bboxes)
            print(f"[Candidates] landscape: using saliency top_bboxes only, count={len(final_candidates)}")
            
        complete_boxes = []
        for bbox in final_candidates:
            safety = self._safety_gate(bbox, human_boxes, object_boxes, scene_type)
            if safety["pass"]:
                complete_boxes.append(bbox)

        if not complete_boxes:
            complete_boxes = final_candidates

        scored_boxes = self.score_with_fusion(img, complete_boxes, scene_type=scene_type)
        top10 = scored_boxes[:top_k]
        

        if not top10:
            return {
                "config": self.config.to_dict(),
                "scene_type": scene_type,
                "scene_stats": scene_stats,
                "top10": [],
                "top5": [],
                "saliency_map": sal_map,
                "human_boxes": human_boxes,
                "object_boxes": object_boxes,
            }

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cropped_images = [img_rgb[item["bbox"].y1:item["bbox"].y2, item["bbox"].x1:item["bbox"].x2] for item in top10]
        clip_scores = self.clip_scorer.score_batch(cropped_images)
        clip_min, clip_max = min(clip_scores), max(clip_scores)
        clip_range = clip_max - clip_min

        if clip_range > 0.04:
            target_range = 0.008
        elif clip_range > 0.025:
            target_range = 0.012
        elif clip_range > 0.015:
            target_range = 0.016
        else:
            target_range = clip_range

        if clip_range > target_range and clip_range > 0:
            compress_ratio = target_range / clip_range
            clip_scores_smoothed = [clip_min + (s - clip_min) * compress_ratio for s in clip_scores]
        else:
            clip_scores_smoothed = clip_scores

        min_s, max_s = min(clip_scores_smoothed), max(clip_scores_smoothed)
        if max_s > min_s:
            clip_scores_norm = [(s - min_s) / (max_s - min_s) for s in clip_scores_smoothed]
            clip_scores_norm = [0.45 + x * 0.1 for x in clip_scores_norm]
        else:
            clip_scores_norm = [0.50] * len(clip_scores)

        if clip_range > 0.035:
            clip_weight = 0.05
            fusion_weight = 0.95
        elif clip_range > 0.02:
            clip_weight = 0.10
            fusion_weight = 0.90
        elif clip_range > 0.01:
            clip_weight = 0.15
            fusion_weight = 0.85
        else:
            clip_weight = 0.20
            fusion_weight = 0.80

        for i, item in enumerate(top10):
            item["clip_score_raw"] = clip_scores[i]
            item["clip_score_smoothed"] = clip_scores_smoothed[i]
            item["clip_score_norm"] = clip_scores_norm[i]
            item["clip_weight"] = clip_weight
            item["fusion_weight"] = fusion_weight
            item["rerank_score"] = clip_weight * clip_scores_norm[i] + fusion_weight * item["fusion_score"]

        top10.sort(key=lambda x: x["rerank_score"], reverse=True)
        top5 = top10[:5]
        
        return {
            "config": self.config.to_dict(),
            "scene_type": scene_type,
            "scene_stats": scene_stats,
            "top10": top10,
            "top5": top5,
            "saliency_map": sal_map,
            "human_boxes": human_boxes,
            "object_boxes": object_boxes,
        }

    def _safety_gate(self, candidate_bbox: BBox, human_boxes, object_boxes, scene_type: str):
        human_cov = self._max_coverage(candidate_bbox, human_boxes)
        object_cov = self._max_coverage(candidate_bbox, object_boxes)

        touch_human = self._edge_touch_ratio(candidate_bbox, human_boxes)
        touch_object = self._edge_touch_ratio(candidate_bbox, object_boxes)
        edge_touch = max(touch_human, touch_object)
        edge_ok = edge_touch <= self.config.completeness.max_edge_touch_ratio

        if scene_type == "portrait":
            human_ok = (not human_boxes) or (human_cov >= self.config.completeness.human_coverage_threshold)
            object_ok = (not object_boxes) or (object_cov >= self.config.completeness.object_coverage_threshold)
            return {
                "pass": human_ok and object_ok and edge_ok,
                "human_cov": human_cov,
                "object_cov": object_cov,
                "edge_touch": edge_touch,
            }
        
        if scene_type == "subject":
            object_ok = (not object_boxes) or (
                object_cov >= self.config.completeness.object_coverage_threshold * 0.5
            )
            return {
                "pass": object_ok and edge_ok,
                "human_cov": human_cov,
                "object_cov": object_cov,
                "edge_touch": edge_touch,
            }
        

        #structure_score = 0.0
        # 上面这句仅为了保持接口占位，真正分数在 score_with_fusion 中计算；这里不重复耗时，沿用结构几何门控。
        crop_ok = self._structure_gate(candidate_bbox, human_boxes, object_boxes)
        return {
            "pass": crop_ok and edge_ok,
            "human_cov": human_cov,
            "object_cov": object_cov,
            "edge_touch": edge_touch,
            "structure_score": 0.0,
        }

    def _structure_gate(self, candidate_bbox: BBox, human_boxes, object_boxes):
        # 仅对非人像场景启用：保证结构候选不会太碎、太偏、太贴边
        if candidate_bbox.area <= 0:
            return False
        border_margin = self.config.completeness.edge_margin_px
        img_like = max(candidate_bbox.width, candidate_bbox.height)
        border_ok = candidate_bbox.x1 > border_margin or candidate_bbox.y1 > border_margin or img_like > border_margin

        aspect = candidate_bbox.width / max(1, candidate_bbox.height)
        aspect_ok = 0.18 <= aspect <= 5.5
        size_ok = candidate_bbox.area >= max(1200, self.config.completeness.min_area_ratio * 0.5)

        return border_ok and aspect_ok and size_ok
    
    def _iou(self, a: BBox, b: BBox) -> float:
        """计算两个 BBox 的交并比 (IoU)"""
        ix1 = max(a.x1, b.x1)
        iy1 = max(a.y1, b.y1)
        ix2 = min(a.x2, b.x2)
        iy2 = min(a.y2, b.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = a.area + b.area - inter
        return inter / max(union, 1)
