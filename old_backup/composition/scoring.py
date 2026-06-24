# composition/scoring.py
"""
智能取景核心评分模块
封装了显著性筛选、人体/物体检测、美学评分、综合排序、框扩展和面积约束。
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from crop.bbox_utils import BBox
from saliency.detector import FTDetector
from saliency.saliency_utils import filter_top_bboxes_by_percentile, get_subject_center
from composition.aesthetic_scorer import AestheticScorer
from composition.human_detector import HumanDetector
from composition.object_detector import ObjectDetector

class AestheticPipeline:
    """
    智能取景完整流程：
    1. 生成候选框（由队友的 generate_candidates 提供）
    2. 显著性筛选前 top_percent_sal
    3. 同时进行人体检测（YOLO + MTCNN 兜底）和物体检测（YOLO）
    4. 对每个候选框计算综合得分（美学 + 人体覆盖 + 物体覆盖 + 中心得分）
    5. 排序、保底逻辑、扩展最佳框、面积约束
    6. 返回最终候选框列表和对应得分
    """
    def __init__(self,
                 top_percent_sal: float = 0.3,
                 top_percent_aes: float = 0.05,
                 w_human: float = 1.0,
                 w_object: float = 0.8,
                 w_center: float = 0.2,
                 min_cover_thresh: float = 0.3,
                 expand_ratio: float = 0.15,
                 target_area_ratio: float = 0.2,
                 area_low: float = 0.2,
                 area_high: float = 0.5,
                 yolo_conf_thresh: float = 0.25):
        """
        参数说明：
        - top_percent_sal: 显著性筛选保留前百分之几的候选框
        - top_percent_aes: 最终保留前百分之几的候选框（按综合得分）
        - w_human: 人体覆盖得分权重
        - w_object: 物体覆盖得分权重
        - w_center: 中心得分权重
        - min_cover_thresh: 保底阈值，若第一名覆盖度低于此值则强制替换为覆盖度最高的框
        - expand_ratio: 最佳框扩展比例（相对于主体框）
        - target_area_ratio: 面积约束目标比例（最终框占原图面积的比例）
        - area_low: 低于此比例触发面积约束
        - area_high: 高于此比例触发面积约束
        - yolo_conf_thresh: YOLO 检测置信度阈值
        """
        self.top_percent_sal = top_percent_sal
        self.top_percent_aes = top_percent_aes
        self.w_human = w_human
        self.w_object = w_object
        self.w_center = w_center
        self.min_cover_thresh = min_cover_thresh
        self.expand_ratio = expand_ratio
        self.target_area_ratio = target_area_ratio
        self.area_low = area_low
        self.area_high = area_high

        # 初始化各个子模块
        self.saliency_detector = FTDetector()
        self.aesthetic_scorer = AestheticScorer()
        self.mtcnn_detector = HumanDetector()
        self.yolo = ObjectDetector(conf_threshold=yolo_conf_thresh)

    def _compute_iou(self, box1: Tuple[int,int,int,int], box2: Tuple[int,int,int,int]) -> float:
        x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def _expand_bbox(self, bbox: BBox, subject_box: Tuple[int,int,int,int], img_w: int, img_h: int) -> BBox:
        """扩展候选框以包含主体框，并向外扩展 expand_ratio"""
        x1 = min(bbox.x1, subject_box[0]); y1 = min(bbox.y1, subject_box[1])
        x2 = max(bbox.x2, subject_box[2]); y2 = max(bbox.y2, subject_box[3])
        w = x2 - x1; h = y2 - y1
        dx = int(w * self.expand_ratio); dy = int(h * self.expand_ratio)
        x1 = max(0, x1 - dx); y1 = max(0, y1 - dy)
        x2 = min(img_w, x2 + dx); y2 = min(img_h, y2 + dy)
        return BBox(x1, y1, x2, y2, bbox.scale)

    def _resize_bbox_to_target_area(self, bbox: BBox, img_w: int, img_h: int) -> BBox:
        """将框缩放至目标面积比例，保持中心点不变"""
        area_ratio = bbox.area / (img_w * img_h)
        if abs(area_ratio - self.target_area_ratio) < 0.01:
            return bbox
        scale = np.sqrt(self.target_area_ratio / area_ratio)
        new_w = int(bbox.width * scale)
        new_h = int(bbox.height * scale)
        cx = (bbox.x1 + bbox.x2) / 2
        cy = (bbox.y1 + bbox.y2) / 2
        x1 = int(cx - new_w/2)
        y1 = int(cy - new_h/2)
        x2 = x1 + new_w
        y2 = y1 + new_h
        # 边界裁剪
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > img_w:
            x1 -= (x2 - img_w)
            x2 = img_w
        if y2 > img_h:
            y1 -= (y2 - img_h)
            y2 = img_h
        return BBox(x1, y1, x2, y2, bbox.scale)

    def process(self, image: np.ndarray, candidates: List[BBox]):
        """
        主流程：输入 BGR 图像和候选框列表，返回 (final_bboxes, final_scores, human_boxes, object_boxes, sal_map)
        - final_bboxes: 最终选出的前 top_percent_aes 个 BBox 对象
        - final_scores: 对应的综合得分
        - human_boxes: 检测到的人体框列表（用于可视化）
        - object_boxes: 检测到的物体框列表（用于可视化）
        - sal_map: 显著图（用于可视化）
        """
        h, w = image.shape[:2]
        total_area = w * h

        # 1. 显著性检测和筛选
        sal_map = self.saliency_detector.detect(image)
        top_bboxes, _, _ = filter_top_bboxes_by_percentile(sal_map, candidates, top_percent=self.top_percent_sal)
        if not top_bboxes:
            return [], [], [], [], sal_map

        # 2. 人体和物体检测
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        human_boxes, object_boxes = self.yolo.detect_all(img_rgb, verbose=False)
        # 人体兜底：若 YOLO 未检测到人体，尝试 MTCNN
        if not human_boxes:
            human_boxes = self.mtcnn_detector.detect_human_bboxes(img_rgb)

        # 3. 确定扩展用的主体框（优先人体，否则最大物体）
        expand_box = None
        if human_boxes:
            expand_box = max(human_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        elif object_boxes:
            expand_box = max(object_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))

        # 4. 目标中心（用于中心得分）
        if expand_box:
            target_cx = (expand_box[0] + expand_box[2]) / 2
            target_cy = (expand_box[1] + expand_box[3]) / 2
        else:
            target_cx, target_cy = get_subject_center(sal_map)

        # 5. 计算每个候选框的综合得分
        scored = []
        for bbox in top_bboxes:
            aes = self.aesthetic_scorer.get_score_for_bbox(image, bbox)
            human_cover = max([self._compute_iou((bbox.x1, bbox.y1, bbox.x2, bbox.y2), hb) for hb in human_boxes]) if human_boxes else 0.0
            obj_cover = max([self._compute_iou((bbox.x1, bbox.y1, bbox.x2, bbox.y2), ob) for ob in object_boxes]) if object_boxes else 0.0
            cover = self.w_human * human_cover + self.w_object * obj_cover
            cx = (bbox.x1 + bbox.x2) / 2
            cy = (bbox.y1 + bbox.y2) / 2
            dist = np.hypot(cx - target_cx, cy - target_cy)
            max_dist = np.hypot(w, h) / 2
            center_score = 1.0 - min(1.0, dist / max_dist)
            final = aes + cover + self.w_center * center_score
            scored.append((bbox, final, aes, cover, center_score, human_cover, obj_cover))

        scored.sort(key=lambda x: x[1], reverse=True)

        # 6. 保底逻辑：若第一名覆盖度低于阈值，则用覆盖度最高的框替换
        if expand_box:
            best_cover = max(scored, key=lambda x: x[3])  # x[3] 是 cover 得分
            if scored[0][3] < self.min_cover_thresh and best_cover[3] > self.min_cover_thresh:
                scored.remove(best_cover)
                scored.insert(0, best_cover)

        # 7. 取前 top_percent_aes 个框
        k = max(1, int(len(scored) * self.top_percent_aes))
        final_bboxes_raw = [x[0] for x in scored[:k]]
        final_scores_raw = [x[1] for x in scored[:k]]

        # 8. 扩展最佳框 + 面积约束
        final_bboxes = []
        final_scores = []
        for idx, (bbox, score) in enumerate(zip(final_bboxes_raw, final_scores_raw)):
            if idx == 0 and expand_box:
                if self._compute_iou((bbox.x1, bbox.y1, bbox.x2, bbox.y2), expand_box) > 0.1:
                    bbox = self._expand_bbox(bbox, expand_box, w, h)
            # 面积约束（仅对最佳框）
            if idx == 0:
                area_ratio = bbox.area / total_area
                if area_ratio < self.area_low or area_ratio > self.area_high:
                    bbox = self._resize_bbox_to_target_area(bbox, w, h)
            final_bboxes.append(bbox)
            final_scores.append(score)

        return final_bboxes, final_scores, human_boxes, object_boxes, sal_map