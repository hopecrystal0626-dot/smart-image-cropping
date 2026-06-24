# -*- coding: utf-8 -*-

"""后台加载线程，调用 pipeline 处理图片"""

import traceback
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np

from smart_framing.core.pipeline import process_image


class LoadWorker(QThread):
    progress = pyqtSignal(int, str)                # 进度值, 状态文字
    finished = pyqtSignal(object, object, object, object)  # candidates, img_bgr, depth_pixmap, seg_pixmap
    error = pyqtSignal(str)

    def __init__(self, img_path, use_depth=True):
        super().__init__()
        self.img_path = img_path
        self.use_depth = use_depth

    def run(self):
        try:
            def progress_callback(value, msg):
                self.progress.emit(value, msg)

            result = process_image(
                str(self.img_path),
                use_depth=self.use_depth,
                save_vis=False,
                progress_callback=progress_callback
            )

            img_rgb = result['image_rgb']
            ranked = result['all_records']

            # 转换为界面需要的 candidates 格式
            candidates_list = []
            for r in ranked:
                b = r["box"]
                candidates_list.append({
                    'x1': int(b.x1),
                    'y1': int(b.y1),
                    'x2': int(b.x2),
                    'y2': int(b.y2),
                    'final_score': float(r.get('final_score', 0.0)),
                    'aes_norm': float(r.get('aes_norm', 0.0)),
                    'content_score': float(r.get('content_score', 0.0)),
                    'thirds_score': float(r.get('thirds_score', 0.0)),
                    'center_score': float(r.get('center_score', 0.0)),
                    'object_clip_penalty': float(r.get('object_clip_penalty', 0.0)),
                    'missing_subject': float(r.get('missing_subject', 0.0)),
                    'depth_score': float(r.get('depth_score', 0.0)),
                })

            depth_vis = result.get('depth_vis')
            seg_vis = result.get('seg_vis')

            depth_pixmap = None
            if depth_vis is not None:
                h, w, ch = depth_vis.shape
                bytes_per_line = ch * w
                qimg = QImage(depth_vis.data, w, h, bytes_per_line, QImage.Format_RGB888)
                depth_pixmap = QPixmap.fromImage(qimg)

            seg_pixmap = None
            if seg_vis is not None:
                h, w, ch = seg_vis.shape
                bytes_per_line = ch * w
                qimg = QImage(seg_vis.data, w, h, bytes_per_line, QImage.Format_RGB888)
                seg_pixmap = QPixmap.fromImage(qimg)

            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            self.progress.emit(100, "处理完成")
            self.finished.emit(candidates_list, img_bgr, depth_pixmap, seg_pixmap)

        except Exception as e:
            traceback.print_exc()
            self.error.emit(traceback.format_exc())