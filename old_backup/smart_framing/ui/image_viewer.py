# -*- coding: utf-8 -*-
"""
ImageViewer 图像显示控件（支持拖拽取景框、加载文字）
"""
import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush, QFont, QCursor


class ImageViewer(QWidget):
    bbox_changed = pyqtSignal(dict)
    MODE_NONE = 0
    MODE_MOVE = 1
    MODE_RESIZE_TL = 2
    MODE_RESIZE_TR = 3
    MODE_RESIZE_BL = 4
    MODE_RESIZE_BR = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
        self.scale = 1.0
        self.offset = QPoint(0, 0)
        self.bbox = None
        self.aspect_ratio = None
        self.current_font = QFont("宋体", 10)
        self.loading_text = ""

        self.drag_mode = self.MODE_NONE
        self.drag_start = QPoint()
        self.bbox_start = None

        self.handle_radius = 8
        self.line_color = QColor(53, 80, 163)
        self.highlight_color = QColor(24, 169, 153, 200)

        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setStyleSheet("""
            ImageViewer {
                background: #F5F7FB;
                border: 2px solid #D7E7FA;
                border-radius: 12px;
            }
        """)

    def set_loading_text(self, text):
        self.loading_text = text
        self.update()

    def set_font(self, font):
        self.current_font = font
        self.update()

    def set_image(self, img_bgr):
        if img_bgr is None:
            self.pixmap = None
            self.update()
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qt_img)
        self.update_scale()

    def set_bbox(self, bbox_dict):
        self.bbox = bbox_dict
        self.update()

    def set_aspect_ratio(self, ratio_w, ratio_h):
        if ratio_w is not None and ratio_w > 0 and ratio_h > 0:
            self.aspect_ratio = (ratio_w, ratio_h)
        else:
            self.aspect_ratio = None

    def enforce_aspect_ratio(self, bbox_dict):
        if self.aspect_ratio is None or self.pixmap is None:
            return bbox_dict
        ratio_w, ratio_h = self.aspect_ratio
        x1, y1, x2, y2 = bbox_dict['x1'], bbox_dict['y1'], bbox_dict['x2'], bbox_dict['y2']
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        target_ratio = ratio_w / ratio_h
        if w <= 0 or h <= 0:
            return bbox_dict
        if w / h > target_ratio:
            new_h = w / target_ratio
            new_w = w
        else:
            new_w = h * target_ratio
            new_h = h
        new_x1 = int(cx - new_w / 2)
        new_y1 = int(cy - new_h / 2)
        new_x2 = int(cx + new_w / 2)
        new_y2 = int(cy + new_h / 2)
        img_w = self.pixmap.width()
        img_h = self.pixmap.height()
        if new_x1 < 0:
            new_x2 -= new_x1
            new_x1 = 0
        if new_y1 < 0:
            new_y2 -= new_y1
            new_y1 = 0
        if new_x2 > img_w:
            new_x1 -= (new_x2 - img_w)
            new_x2 = img_w
        if new_y2 > img_h:
            new_y1 -= (new_y2 - img_h)
            new_y2 = img_h
        if new_x2 - new_x1 < 20:
            new_x1 = max(0, int(cx - 10))
            new_x2 = min(img_w, int(cx + 10))
        if new_y2 - new_y1 < 20:
            new_y1 = max(0, int(cy - 10))
            new_y2 = min(img_h, int(cy + 10))
        return {'x1': int(new_x1), 'y1': int(new_y1), 'x2': int(new_x2), 'y2': int(new_y2)}

    def update_scale(self):
        if self.pixmap is None:
            return
        widget_size = self.size()
        img_size = self.pixmap.size()
        if img_size.width() <= 0 or img_size.height() <= 0:
            return
        scale_w = widget_size.width() / img_size.width()
        scale_h = widget_size.height() / img_size.height()
        self.scale = min(scale_w, scale_h) * 0.8
        self.offset = QPoint(0, 0)
        self.update()

    # ---------- 坐标转换 ----------
    def image_to_widget(self, x_img, y_img):
        if self.pixmap is None:
            return x_img, y_img
        img_w = self.pixmap.width()
        img_h = self.pixmap.height()
        disp_w = int(img_w * self.scale)
        disp_h = int(img_h * self.scale)
        ox = (self.width() - disp_w) // 2 + self.offset.x()
        oy = (self.height() - disp_h) // 2 + self.offset.y()
        wx = ox + x_img * self.scale
        wy = oy + y_img * self.scale
        return wx, wy

    def widget_to_image(self, wx, wy):
        if self.pixmap is None:
            return wx, wy
        img_w = self.pixmap.width()
        img_h = self.pixmap.height()
        disp_w = int(img_w * self.scale)
        disp_h = int(img_h * self.scale)
        ox = (self.width() - disp_w) // 2 + self.offset.x()
        oy = (self.height() - disp_h) // 2 + self.offset.y()
        xi = (wx - ox) / self.scale
        yi = (wy - oy) / self.scale
        return xi, yi

    def get_handle_at(self, wx, wy):
        if self.bbox is None or self.pixmap is None:
            return self.MODE_NONE
        x1, y1, x2, y2 = self.bbox['x1'], self.bbox['y1'], self.bbox['x2'], self.bbox['y2']
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        for i, (cx, cy) in enumerate(corners):
            wx_c, wy_c = self.image_to_widget(cx, cy)
            if abs(wx - wx_c) <= self.handle_radius and abs(wy - wy_c) <= self.handle_radius:
                return [self.MODE_RESIZE_TL, self.MODE_RESIZE_TR,
                        self.MODE_RESIZE_BL, self.MODE_RESIZE_BR][i]
        xi, yi = self.widget_to_image(wx, wy)
        if x1 <= xi <= x2 and y1 <= yi <= y2:
            return self.MODE_MOVE
        return self.MODE_NONE

    def update_cursor(self, wx, wy):
        mode = self.get_handle_at(wx, wy)
        if mode in (self.MODE_RESIZE_TL, self.MODE_RESIZE_BR):
            self.setCursor(QCursor(Qt.SizeFDiagCursor))
        elif mode in (self.MODE_RESIZE_TR, self.MODE_RESIZE_BL):
            self.setCursor(QCursor(Qt.SizeBDiagCursor))
        elif mode == self.MODE_MOVE:
            self.setCursor(QCursor(Qt.SizeAllCursor))
        else:
            self.setCursor(QCursor(Qt.ArrowCursor))

    def mousePressEvent(self, event):
        if self.pixmap is None or self.bbox is None:
            return
        if event.button() == Qt.LeftButton:
            wx, wy = event.x(), event.y()
            mode = self.get_handle_at(wx, wy)
            if mode != self.MODE_NONE:
                self.drag_mode = mode
                self.drag_start = QPoint(wx, wy)
                self.bbox_start = self.bbox.copy()
                self.setMouseTracking(True)

    def mouseMoveEvent(self, event):
        wx, wy = event.x(), event.y()
        self.update_cursor(wx, wy)
        if self.drag_mode == self.MODE_NONE or self.bbox_start is None:
            return
        dx = (wx - self.drag_start.x()) / self.scale
        dy = (wy - self.drag_start.y()) / self.scale
        new_bbox = self.bbox_start.copy()
        if self.drag_mode == self.MODE_MOVE:
            new_bbox['x1'] += dx
            new_bbox['y1'] += dy
            new_bbox['x2'] += dx
            new_bbox['y2'] += dy
        else:
            if self.drag_mode in (self.MODE_RESIZE_TL, self.MODE_RESIZE_BL):
                new_bbox['x1'] += dx
            if self.drag_mode in (self.MODE_RESIZE_TR, self.MODE_RESIZE_BR):
                new_bbox['x2'] += dx
            if self.drag_mode in (self.MODE_RESIZE_TL, self.MODE_RESIZE_TR):
                new_bbox['y1'] += dy
            if self.drag_mode in (self.MODE_RESIZE_BL, self.MODE_RESIZE_BR):
                new_bbox['y2'] += dy
        img_w = self.pixmap.width()
        img_h = self.pixmap.height()
        new_bbox['x1'] = int(max(0, min(new_bbox['x1'], img_w - 20)))
        new_bbox['y1'] = int(max(0, min(new_bbox['y1'], img_h - 20)))
        new_bbox['x2'] = int(max(20, min(new_bbox['x2'], img_w)))
        new_bbox['y2'] = int(max(20, min(new_bbox['y2'], img_h)))
        if self.aspect_ratio is not None:
            new_bbox = self.enforce_aspect_ratio(new_bbox)
        self.bbox = new_bbox
        self.update()
        self.bbox_changed.emit(new_bbox)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_mode = self.MODE_NONE
            self.bbox_start = None
            self.setMouseTracking(True)

    def leaveEvent(self, event):
        self.setCursor(QCursor(Qt.ArrowCursor))
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self.pixmap is None:
            painter.fillRect(self.rect(), QColor(245, 247, 251))
            painter.setPen(QColor(43, 54, 70))
            painter.setFont(self.current_font)
            if self.loading_text:
                painter.drawText(self.rect(), Qt.AlignCenter, self.loading_text)
            else:
                painter.drawText(self.rect(), Qt.AlignCenter, "📷 请加载图片")
            return
        painter.fillRect(self.rect(), QColor(245, 247, 251))
        img_w = int(self.pixmap.width() * self.scale)
        img_h = int(self.pixmap.height() * self.scale)
        x = (self.width() - img_w) // 2 + self.offset.x()
        y = (self.height() - img_h) // 2 + self.offset.y()
        painter.drawPixmap(int(x), int(y), int(img_w), int(img_h), self.pixmap)
        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox['x1'], self.bbox['y1'], self.bbox['x2'], self.bbox['y2']
            sx1, sy1 = self.image_to_widget(x1, y1)
            sx2, sy2 = self.image_to_widget(x2, y2)
            sw = sx2 - sx1
            sh = sy2 - sy1
            painter.setBrush(QBrush(QColor(0, 0, 0, 70)))
            painter.setPen(Qt.NoPen)
            painter.drawRect(0, 0, int(self.width()), int(sy1))
            painter.drawRect(0, int(sy2), int(self.width()), int(self.height() - sy2))
            painter.drawRect(0, int(sy1), int(sx1), int(sh))
            painter.drawRect(int(sx2), int(sy1), int(self.width() - sx2), int(sh))
            pen = QPen(self.line_color, 2.5)
            pen.setStyle(Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(int(sx1), int(sy1), int(sw), int(sh))
            pen = QPen(QColor(255, 255, 255, 80), 1, Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(int(sx1 + sw/3), int(sy1), int(sx1 + sw/3), int(sy2))
            painter.drawLine(int(sx1 + 2*sw/3), int(sy1), int(sx1 + 2*sw/3), int(sy2))
            painter.drawLine(int(sx1), int(sy1 + sh/3), int(sx2), int(sy1 + sh/3))
            painter.drawLine(int(sx1), int(sy1 + 2*sh/3), int(sx2), int(sy1 + 2*sh/3))
            painter.setBrush(QBrush(self.highlight_color))
            painter.setPen(QPen(Qt.white, 1.5))
            for cx, cy in [(sx1, sy1), (sx2, sy1), (sx1, sy2), (sx2, sy2)]:
                painter.drawEllipse(QPoint(int(cx), int(cy)), self.handle_radius, self.handle_radius)