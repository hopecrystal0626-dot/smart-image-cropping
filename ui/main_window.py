# -*- coding: utf-8 -*-

"""主窗口"""

import os
import sys
import random
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSplitter, QFileDialog, QMessageBox, QListWidget, QListWidgetItem,
    QGridLayout, QGroupBox, QSizePolicy, QComboBox, QScrollArea, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QFont, QImage, QPixmap

from smart_framing.ui.image_viewer import ImageViewer
from smart_framing.ui.workflow_panel import WorkflowPanel
from smart_framing.ui.loader import LoadWorker
from smart_framing.ui.image_processing import apply_adjustments, apply_aspect_ratio

import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能取景器")
        self.setGeometry(100, 50, 1500, 950)
        self.setStyleSheet(self.get_style())

        # ---- 状态 ----
        self.img_path = None
        self.original_img = None
        self.cropped_img = None
        self.candidates = []
        self.current_idx = 0
        self.top_n = 10

        self.MODE_PREVIEW = 'preview'
        self.MODE_EDIT = 'edit'
        self.MODE_COMPARE = 'compare'
        self.current_mode = self.MODE_PREVIEW

        self.params = {'brightness': 0, 'contrast': 0, 'sharpness': 0,
                       'highlights': 0, 'shadows': 0,
                       'temperature': 0, 'saturation': 0, 'exposure': 0}

        self.load_thread = None

        # 保存原始 pixmap 以便缩放
        self._depth_pixmap_raw = None
        self._seg_pixmap_raw = None

        # ---- 主布局 ----
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # ---- 工具栏 ----
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)
        btn_style = """
            QPushButton {
                background: #F2F6FA;
                border: 1px solid #B8CEE4;
                border-radius: 8px;
                padding: 6px 16px;
                font-weight: 500;
                color: #2C3E50;
            }
            QPushButton:hover {
                background: #E4EEF7;
                border-color: #4A90D9;
            }
            QPushButton:pressed {
                background: #D4E2F0;
            }
            QPushButton:checked {
                background: #4A90D9;
                color: white;
                border-color: #4A90D9;
            }
            QPushButton:disabled {
                background: #E4EEF7;
                color: #9AB5CC;
                border-color: #B8CEE4;
            }
        """
        self.btn_load = QPushButton("📂 加载图片")
        self.btn_load.clicked.connect(self.load_image)
        self.btn_load.setStyleSheet(btn_style)
        self.btn_edit = QPushButton("✏️ 编辑取景")
        self.btn_edit.clicked.connect(self.toggle_edit_mode)
        self.btn_edit.setStyleSheet(btn_style)
        self.btn_confirm = QPushButton("✅ 确认取景")
        self.btn_confirm.clicked.connect(self.confirm_crop)
        self.btn_confirm.setEnabled(False)
        self.btn_confirm.setStyleSheet(btn_style)
        self.btn_compare = QPushButton("🔄 前后对比")
        self.btn_compare.setCheckable(True)
        self.btn_compare.clicked.connect(self.toggle_compare)
        self.btn_compare.setStyleSheet(btn_style)
        self.btn_save = QPushButton("💾 保存结果")
        self.btn_save.clicked.connect(self.save_result)
        self.btn_save.setStyleSheet(btn_style)
        self.lbl_info = QLabel("就绪")
        self.lbl_info.setStyleSheet("color: #7A9BB5;")

        toolbar.addWidget(self.btn_load)
        toolbar.addWidget(self.btn_edit)
        toolbar.addWidget(self.btn_confirm)
        toolbar.addWidget(self.btn_compare)
        toolbar.addWidget(self.btn_save)
        toolbar.addStretch()
        toolbar.addWidget(self.lbl_info)
        main_layout.addLayout(toolbar)

        # ---- 主体水平分割 ----
        main_h_splitter = QSplitter(Qt.Horizontal)
        main_h_splitter.setHandleWidth(8)
        main_h_splitter.setStyleSheet("QSplitter::handle { background: #B8CEE4; border-radius: 4px; }")

        # ----- 左侧大区域 -----
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # 上部分：图片 + 缩略图
        top_h_widget = QWidget()
        top_h_layout = QHBoxLayout(top_h_widget)
        top_h_layout.setContentsMargins(0, 0, 0, 0)
        top_h_layout.setSpacing(10)

        self.image_viewer = ImageViewer()
        self.image_viewer.bbox_changed.connect(self.update_bbox_info)

        # 缩略图面板
        thumb_panel = QWidget()
        thumb_panel.setStyleSheet("""
            background: #F2F6FA;
            border-radius: 12px;
            border: 1px solid #B8CEE4;
        """)
        thumb_layout = QVBoxLayout(thumb_panel)
        thumb_layout.setContentsMargins(0, 0, 0, 0)
        thumb_layout.setSpacing(0)

        thumb_label = QLabel("📋 推荐裁剪（点击切换）")
        thumb_label.setStyleSheet("color: #2C3E50; font-weight: bold; padding: 4px 6px;")
        thumb_layout.addWidget(thumb_label)

        thumb_scroll = QScrollArea()
        thumb_scroll.setWidgetResizable(True)
        thumb_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        thumb_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        thumb_scroll.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical { background: #DCE8F5; width: 8px; border-radius: 4px; }
            QScrollBar::handle:vertical { background: #B8CEE4; border-radius: 4px; }
            QScrollBar::handle:vertical:hover { background: #9AB5CC; }
        """)
        self.thumb_list = QListWidget()
        self.thumb_list.setFlow(QListWidget.TopToBottom)
        self.thumb_list.setWrapping(False)
        self.thumb_list.setResizeMode(QListWidget.Adjust)
        self.thumb_list.setSpacing(6)
        self.thumb_list.setStyleSheet("""
            QListWidget {
                background: transparent;
                border: none;
                outline: none;
                padding: 0;
                margin: 0;
            }
            QListWidget::item {
                background: transparent;
                border: none;
                padding: 2px;
                margin: 4px;
            }
            QListWidget::item:selected {
                background: rgba(74, 144, 217, 0.15);
                border: 2px solid #4A90D9;
                border-radius: 6px;
                padding: 2px;
                margin: 4px;
            }
        """)
        self.thumb_list.itemClicked.connect(self.on_thumb_clicked)
        thumb_scroll.setWidget(self.thumb_list)
        thumb_layout.addWidget(thumb_scroll)

        top_h_layout.addWidget(self.image_viewer, 4)
        top_h_layout.addWidget(thumb_panel, 1)

        left_layout.addWidget(top_h_widget, 3)

        # ----- 暗房面板（含比例选择） -----
        dark_widget = QWidget()
        dark_widget.setStyleSheet("""
            background: #F2F6FA;
            border-radius: 12px;
            border: 1px solid #B8CEE4;
        """)
        dark_layout = QVBoxLayout(dark_widget)
        dark_layout.setContentsMargins(12, 8, 12, 8)
        dark_layout.setSpacing(6)

        dark_title = QLabel("🎨 图像调节")
        dark_title.setStyleSheet("color: #2C3E50; font-weight: bold;")
        dark_layout.addWidget(dark_title)

        grid = QGridLayout()
        grid.setSpacing(6)

        slider_configs = [
            ("brightness", "亮度", 0, -100, 100),
            ("contrast", "对比度", 0, -100, 100),
            ("sharpness", "锐度", 0, -100, 100),
            ("highlights", "高光", 0, -100, 100),
            ("shadows", "阴影", 0, -100, 100),
            ("temperature", "色温", 0, -100, 100),
            ("saturation", "饱和度", 0, -100, 100),
            ("exposure", "曝光", 0, -100, 100)
        ]
        self.sliders = {}
        self.font_update_widgets = [dark_title, thumb_label]
        row, col = 0, 0
        for eng_key, display_name, default, minv, maxv in slider_configs:
            hbox = QHBoxLayout()
            hbox.setSpacing(6)
            name_lbl = QLabel(display_name)
            name_lbl.setFixedWidth(100)
            name_lbl.setStyleSheet("color: #7A9BB5; font-weight: 500;")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(minv, maxv)
            slider.setValue(default)
            slider.setFixedHeight(16)
            slider.setStyleSheet("""
                QSlider::groove:horizontal {
                    height: 4px;
                    background: #B8CEE4;
                    border-radius: 2px;
                }
                QSlider::handle:horizontal {
                    background: #4A90D9;
                    width: 14px;
                    height: 14px;
                    margin: -5px 0;
                    border-radius: 7px;
                    border: 2px solid #FFFFFF;
                }
                QSlider::handle:horizontal:hover {
                    background: #E8A87C;
                }
            """)
            slider.valueChanged.connect(lambda v, k=eng_key: self.on_slider_changed(k, v))
            val_lbl = QLabel(str(default))
            val_lbl.setFixedWidth(80)
            val_lbl.setStyleSheet("color: #2C3E50; font-weight: 500;")
            hbox.addWidget(name_lbl)
            hbox.addWidget(slider, 1)
            hbox.addWidget(val_lbl)
            grid.addLayout(hbox, row, col)
            self.sliders[eng_key] = (slider, val_lbl, name_lbl)
            self.font_update_widgets.extend([name_lbl, val_lbl])
            col += 1
            if col >= 2:
                col = 0
                row += 1

        dark_layout.addLayout(grid)

        # ---- 比例选择（移到暗房底部） ----
        self.ratio_widget = QWidget()
        self.ratio_widget.setVisible(False)
        ratio_layout = QHBoxLayout(self.ratio_widget)
        ratio_layout.setContentsMargins(0, 0, 0, 0)
        ratio_label = QLabel("📐 裁剪比例")
        ratio_label.setStyleSheet("color: #2C3E50; font-weight: bold;")
        self.ratio_combo = QComboBox()
        self.ratio_combo.addItems(["自由", "9:16", "16:9", "3:4", "4:3", "1:1"])
        self.ratio_combo.currentTextChanged.connect(self.on_ratio_changed)
        self.ratio_combo.setStyleSheet("""
            QComboBox {
                background: #F2F6FA;
                border: 1px solid #B8CEE4;
                border-radius: 6px;
                padding: 4px 8px;
                color: #2C3E50;
            }
            QComboBox:hover {
                border-color: #4A90D9;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(no_such_image);
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #7A9BB5;
                margin-right: 4px;
            }
            QComboBox QAbstractItemView {
                background: #F2F6FA;
                color: #2C3E50;
                selection-background-color: #DCE8F5;
            }
        """)
        ratio_layout.addWidget(ratio_label)
        ratio_layout.addWidget(self.ratio_combo)
        dark_layout.addWidget(self.ratio_widget)

        self.btn_reset = QPushButton("↩️ 恢复默认")
        self.btn_reset.clicked.connect(self.reset_params)
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background: #DCE8F5;
                border: none;
                border-radius: 6px;
                padding: 4px 12px;
                color: #2C3E50;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #B8CEE4;
            }
        """)
        dark_layout.addWidget(self.btn_reset, alignment=Qt.AlignRight)
        self.font_update_widgets.extend([self.btn_reset, ratio_label, self.ratio_combo])

        left_layout.addWidget(dark_widget, 1)

        # ----- 右侧面板（新布局） -----
        right_widget = QWidget()
        right_widget.setMinimumWidth(200)
        right_widget.setStyleSheet("""
            background: #F2F6FA;
            border-radius: 12px;
            border: 1px solid #B8CEE4;
            padding: 6px;
        """)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(6, 6, 6, 6)

        # 1. 工作流进度
        self.workflow = WorkflowPanel()
        right_layout.addWidget(self.workflow)

        # 2. 取景框坐标
        self.bbox_group = QGroupBox("取景框坐标")
        bbox_layout = QVBoxLayout()
        self.bbox_label = QLabel("x1: 0  y1: 0  x2: 0  y2: 0\n宽度: 0  高度: 0")
        self.bbox_label.setStyleSheet("color: #2C3E50; font-weight: 500;")
        bbox_layout.addWidget(self.bbox_label)
        self.bbox_group.setLayout(bbox_layout)
        right_layout.addWidget(self.bbox_group)

        # 3. 深度图
        self.depth_group = QGroupBox("深度图")
        depth_layout = QVBoxLayout()
        self.depth_label = QLabel("暂无深度图")
        self.depth_label.setAlignment(Qt.AlignCenter)
        self.depth_label.setStyleSheet("background: #DCE8F5; min-height: 100px; border-radius: 6px;")
        # 不设置 setScaledContents，我们自己管理缩放
        depth_layout.addWidget(self.depth_label)
        self.depth_group.setLayout(depth_layout)
        right_layout.addWidget(self.depth_group)

        # 4. Mask2Former 分割
        self.mask_group = QGroupBox("语义分割图")
        mask_layout = QVBoxLayout()
        self.mask_label = QLabel("暂无分割图")
        self.mask_label.setAlignment(Qt.AlignCenter)
        self.mask_label.setStyleSheet("background: #DCE8F5; min-height: 100px; border-radius: 6px;")
        # 不设置 setScaledContents
        mask_layout.addWidget(self.mask_label)
        self.mask_group.setLayout(mask_layout)
        right_layout.addWidget(self.mask_group)

        # 调整拉伸因子
        right_layout.setStretchFactor(self.workflow, 0)
        right_layout.setStretchFactor(self.bbox_group, 0)
        right_layout.setStretchFactor(self.depth_group, 1)
        right_layout.setStretchFactor(self.mask_group, 1)

        main_h_splitter.addWidget(left_widget)
        main_h_splitter.addWidget(right_widget)
        main_h_splitter.setStretchFactor(0, 2)
        main_h_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(main_h_splitter, 1)

        # ---- 收集字体更新控件 ----
        self.font_update_widgets.extend([
            self.btn_load, self.btn_edit, self.btn_confirm,
            self.btn_compare, self.btn_save, self.lbl_info,
            self.thumb_list, self.bbox_label, self.depth_label,
            self.mask_label, self.btn_reset, self.ratio_combo,
            thumb_label, dark_title, self.bbox_group,
            self.depth_group, self.mask_group, self.workflow
        ])

        # ---- 动态缩放 ----
        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.on_resize_finished)

        self.update_display()
        self.update_fonts()

    def get_style(self):
        return """
            QMainWindow {
                background: #DCE8F5;
            }
            QLabel {
                color: #2C3E50;
            }
            QSplitter::handle {
                background: #B8CEE4;
            }
            QToolTip {
                background: #F2F6FA;
                color: #2C3E50;
                border: 1px solid #B8CEE4;
            }
            QGroupBox {
                border: 1px solid #B8CEE4;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
            }
        """

    def resizeEvent(self, event):
        self.image_viewer.update_scale()
        self.resize_timer.start(150)
        self.update_fonts()
        # 更新深度图和分割图的显示
        self.update_depth_pixmap()
        self.update_seg_pixmap()
        super().resizeEvent(event)

    def update_fonts(self):
        width = self.width()
        new_size = max(10, min(20, int(width * 0.006)))
        font = QFont("宋体", new_size)

        for widget in self.font_update_widgets:
            if widget is not None:
                widget.setFont(font)

        self._set_font_recursive(self.workflow, font)
        self.image_viewer.set_font(font)
        self.update_thumbs()

    def _set_font_recursive(self, widget, font):
        widget.setFont(font)
        for child in widget.children():
            if isinstance(child, QWidget):
                self._set_font_recursive(child, font)

    def on_resize_finished(self):
        self.update_thumbs()

    # ==================== 比例选择 ====================
    def on_ratio_changed(self, text):
        ratio_map = {
            "自由": (None, None),
            "9:16": (9, 16),
            "16:9": (16, 9),
            "3:4": (3, 4),
            "4:3": (4, 3),
            "1:1": (1, 1)
        }
        ratio_w, ratio_h = ratio_map.get(text, (None, None))
        self.image_viewer.set_aspect_ratio(ratio_w, ratio_h)
        if self.candidates and self.current_idx < len(self.candidates):
            if ratio_w is not None and ratio_h is not None:
                bbox = self.candidates[self.current_idx]
                constrained = self.image_viewer.enforce_aspect_ratio(bbox)
                self.candidates[self.current_idx] = constrained
                self.image_viewer.set_bbox(constrained)
                self.update_display()

    # ==================== 核心显示 ====================
    def get_display_image(self):
        if self.original_img is None:
            return None
        if self.current_mode == self.MODE_PREVIEW:
            if self.cropped_img is None:
                return None
            if all(v == 0 for v in self.params.values()):
                return self.cropped_img.copy()
            return apply_adjustments(self.cropped_img, **self.params)
        elif self.current_mode == self.MODE_EDIT:
            return self.original_img
        elif self.current_mode == self.MODE_COMPARE:
            if self.cropped_img is None:
                return self.original_img
            left = self.original_img.copy()
            right = self.cropped_img.copy() if all(v == 0 for v in self.params.values()) else apply_adjustments(
                self.cropped_img, **self.params)
            h = min(left.shape[0], right.shape[0])
            left = cv2.resize(left, (int(left.shape[1] * h / left.shape[0]), h))
            right = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))
            separator = np.full((h, 8, 3), 200, dtype=np.uint8)
            combined = np.hstack([left, separator, right])
            return combined
        return self.original_img

    def update_display(self):
        display_img = self.get_display_image()
        if display_img is None:
            self.image_viewer.set_image(None)
            self.image_viewer.set_bbox(None)
            return
        self.image_viewer.set_image(display_img)
        if self.current_mode == self.MODE_EDIT:
            if self.candidates and self.current_idx < len(self.candidates):
                self.image_viewer.set_bbox(self.candidates[self.current_idx])
            else:
                self.image_viewer.set_bbox(None)
        else:
            self.image_viewer.set_bbox(None)

        if self.candidates and self.current_idx < len(self.candidates):
            self.update_bbox_info(self.candidates[self.current_idx])
        else:
            self.update_bbox_info(None)

        self.highlight_thumb(self.current_idx)

    # ==================== 坐标信息更新 ====================
    def update_bbox_info(self, bbox):
        if bbox is None:
            self.bbox_label.setText("x1: 0  y1: 0  x2: 0  y2: 0\n宽度: 0  高度: 0")
            return
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        w = x2 - x1
        h = y2 - y1
        self.bbox_label.setText(f"x1: {x1}  y1: {y1}  x2: {x2}  y2: {y2}\n宽度: {w}  高度: {h}")

    # ==================== 加载图片（真实进度） ====================
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择场景图", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if not file_path:
            return

        # ---- 重置所有状态 ----
        self.original_img = None
        self.cropped_img = None
        self.candidates = []
        self.current_idx = 0
        self._depth_pixmap_raw = None
        self._seg_pixmap_raw = None
        self.depth_label.setText("暂无深度图")
        self.mask_label.setText("暂无分割图")
        self.thumb_list.clear()
        self.image_viewer.set_image(None)
        self.image_viewer.set_bbox(None)
        self.bbox_label.setText("x1: 0  y1: 0  x2: 0  y2: 0\n宽度: 0  高度: 0")

        # ---- 恢复暗房参数 ----
        self.reset_params(silent=True)
        for eng_key in self.params.keys():
            if eng_key in self.sliders:
                slider, val_lbl, name_lbl = self.sliders[eng_key]
                slider.blockSignals(True)
                slider.setValue(0)
                slider.blockSignals(False)
                val_lbl.setText("0")
        self.params = {k: 0 for k in self.params}

        # ---- 显示加载状态 ----
        self.img_path = Path(file_path)
        self.image_viewer.set_loading_text("⏳ 正在加载...")
        self.lbl_info.setText("⏳ 正在加载图片，请稍候...")
        self.btn_load.setEnabled(False)
        self.btn_edit.setEnabled(False)
        self.btn_confirm.setEnabled(False)
        self.btn_compare.setEnabled(False)
        self.btn_save.setEnabled(False)

        self.workflow.reset()
        self.workflow.progress_bar.setValue(0)
        self.workflow.status_label.setText("开始加载...")
        self.repaint()

        # ---- 启动后台线程 ----
        self.load_thread = LoadWorker(self.img_path, use_depth=True)
        self.load_thread.progress.connect(self.on_load_progress)
        self.load_thread.finished.connect(self.on_load_finished)
        self.load_thread.error.connect(self.on_load_error)
        self.load_thread.start()

    def on_load_progress(self, value, msg):
        self.workflow.progress_bar.setValue(value)
        self.workflow.status_label.setText(msg)
        if value < 15:
            step = 0
        elif value < 30:
            step = 1
        elif value < 55:
            step = 2
        elif value < 75:
            step = 3
        else:
            step = 4
        self.workflow.update_step(step, msg, value)

    def on_load_finished(self, candidates, img_bgr, depth_pixmap, seg_pixmap):
        self.workflow.progress_bar.setValue(100)
        self.workflow.update_step(5, "处理完成", 100)

        self.original_img = img_bgr
        self.candidates = candidates
        self.current_idx = 0

        if candidates and len(candidates) > 0:
            best = candidates[0]
            x1, y1, x2, y2 = int(best['x1']), int(best['y1']), int(best['x2']), int(best['y2'])
            self.cropped_img = self.original_img[y1:y2, x1:x2]
        else:
            self.cropped_img = None

        # 保存原始 pixmap
        self._depth_pixmap_raw = depth_pixmap
        self._seg_pixmap_raw = seg_pixmap

        # 更新显示
        self.update_depth_pixmap()
        self.update_seg_pixmap()

        self.image_viewer.set_loading_text("")
        self.image_viewer.set_image(self.original_img)
        self.current_mode = self.MODE_PREVIEW
        self.btn_edit.setText("✏️ 编辑取景")
        self.btn_confirm.setEnabled(False)
        self.btn_compare.setChecked(False)
        self.ratio_widget.setVisible(False)

        self.update_thumbs()
        self.update_display()
        self.lbl_info.setText(f"加载完成: {self.img_path.name}  (候选: {len(self.candidates)})")

        self.btn_load.setEnabled(True)
        self.btn_edit.setEnabled(True)
        self.btn_compare.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.load_thread = None

    def on_load_error(self, error_msg):
        QMessageBox.critical(self, "算法错误", f"调用失败：\n{error_msg}")
        self.workflow.reset()
        self.image_viewer.set_loading_text("")
        self.btn_load.setEnabled(True)
        self.btn_edit.setEnabled(True)
        self.btn_compare.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.btn_confirm.setEnabled(False)
        self.lbl_info.setText("加载失败")
        self.load_thread = None

    # ==================== 深度图和分割图缩放更新 ====================
    def update_depth_pixmap(self):
        if self._depth_pixmap_raw is None:
            self.depth_label.setText("暂无深度图")
            return
        # 获取 label 可用大小
        label_size = self.depth_label.size()
        if label_size.width() <= 1 or label_size.height() <= 1:
            return
        # 缩放并居中
        scaled = self._depth_pixmap_raw.scaled(
            label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.depth_label.setPixmap(scaled)
        self.depth_label.setAlignment(Qt.AlignCenter)

    def update_seg_pixmap(self):
        if self._seg_pixmap_raw is None:
            self.mask_label.setText("暂无分割图")
            return
        label_size = self.mask_label.size()
        if label_size.width() <= 1 or label_size.height() <= 1:
            return
        scaled = self._seg_pixmap_raw.scaled(
            label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.mask_label.setPixmap(scaled)
        self.mask_label.setAlignment(Qt.AlignCenter)

    # ==================== 重置参数 ====================
    def reset_params(self, silent=False):
        for eng_key in self.params.keys():
            self.params[eng_key] = 0
            if eng_key in self.sliders:
                slider, val_lbl, name_lbl = self.sliders[eng_key]
                slider.blockSignals(True)
                slider.setValue(0)
                slider.blockSignals(False)
                val_lbl.setText("0")
        self.update_display()
        if not silent:
            self.lbl_info.setText("已恢复原始参数")
            QMessageBox.information(self, "恢复默认", "所有参数已恢复为 0（无调色）")

    # ==================== 编辑模式切换 ====================
    def toggle_edit_mode(self):
        if self.original_img is None:
            return
        if self.current_mode == self.MODE_EDIT:
            self.current_mode = self.MODE_PREVIEW
            self.btn_edit.setText("✏️ 编辑取景")
            self.btn_confirm.setEnabled(False)
            self.btn_compare.setChecked(False)
            self.image_viewer.set_aspect_ratio(None, None)
            self.ratio_combo.setCurrentText("自由")
            self.ratio_widget.setVisible(False)
        else:
            self.current_mode = self.MODE_EDIT
            self.btn_edit.setText("🔙 取消编辑")
            self.btn_confirm.setEnabled(True)
            self.btn_compare.setChecked(False)
            self.ratio_widget.setVisible(True)
            if self.candidates and self.current_idx < len(self.candidates):
                ratio_w, ratio_h = self.image_viewer.aspect_ratio or (None, None)
                if ratio_w is not None and ratio_h is not None:
                    bbox = self.candidates[self.current_idx]
                    constrained = self.image_viewer.enforce_aspect_ratio(bbox)
                    self.candidates[self.current_idx] = constrained
        self.update_display()

    # ==================== 确认取景 ====================
    def confirm_crop(self):
        if self.original_img is None:
            QMessageBox.warning(self, "提示", "请先加载图片")
            return

        bbox = self.image_viewer.bbox
        if bbox is None:
            QMessageBox.warning(self, "提示", "当前没有取景框，请先编辑取景")
            return

        if self.image_viewer.aspect_ratio is not None:
            ratio_w, ratio_h = self.image_viewer.aspect_ratio
            bbox = apply_aspect_ratio(bbox, self.original_img.shape[1], self.original_img.shape[0],
                                      ratio_w, ratio_h)

        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        if x2 <= x1 or y2 <= y1:
            QMessageBox.warning(self, "错误", "裁剪区域无效（宽度或高度为0）")
            return

        self.cropped_img = self.original_img[y1:y2, x1:x2]
        if self.cropped_img.size == 0:
            QMessageBox.warning(self, "错误", "裁剪区域为空，请检查取景框位置")
            return

        self.reset_params(silent=True)
        self.current_mode = self.MODE_PREVIEW
        self.btn_edit.setText("✏️ 编辑取景")
        self.btn_confirm.setEnabled(False)
        self.btn_compare.setChecked(False)
        self.ratio_widget.setVisible(False)
        self.image_viewer.set_aspect_ratio(None, None)
        self.ratio_combo.setCurrentText("自由")

        self.update_display()
        self.lbl_info.setText("已确认取景（自定义裁剪）")
        QMessageBox.information(self, "成功", "裁剪已应用")

    def on_bbox_dragged(self, new_bbox):
        if self.candidates and self.current_idx < len(self.candidates):
            new_bbox['x1'] = int(new_bbox['x1'])
            new_bbox['y1'] = int(new_bbox['y1'])
            new_bbox['x2'] = int(new_bbox['x2'])
            new_bbox['y2'] = int(new_bbox['y2'])
            old = self.candidates[self.current_idx]
            old.update({k: new_bbox[k] for k in ('x1', 'y1', 'x2', 'y2')})
            delta = random.uniform(-0.05, 0.05)
            old['final_score'] = max(0, min(1, old.get('final_score', 0.5) + delta))
            self.update_display()

    def toggle_compare(self, checked):
        if self.original_img is None:
            self.btn_compare.setChecked(False)
            return
        self.current_mode = self.MODE_COMPARE if checked else self.MODE_PREVIEW
        if checked:
            self.btn_edit.setText("✏️ 编辑取景")
            self.btn_confirm.setEnabled(False)
            self.ratio_widget.setVisible(False)
        self.update_display()

    def on_slider_changed(self, eng_key, value):
        self.params[eng_key] = value
        if eng_key in self.sliders:
            slider, val_lbl, name_lbl = self.sliders[eng_key]
            val_lbl.setText(str(value))
        if self.current_mode in [self.MODE_PREVIEW, self.MODE_COMPARE]:
            self.update_display()

    def save_result(self):
        if self.cropped_img is None:
            QMessageBox.warning(self, "提示", "没有可保存的裁剪结果")
            return
        result_img = self.cropped_img.copy() if all(v == 0 for v in self.params.values()) else apply_adjustments(
            self.cropped_img, **self.params)
        if result_img is None:
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存裁剪结果", str(self.img_path.stem) + "_crop.jpg",
            "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if save_path:
            cv2.imwrite(save_path, result_img)
            self.lbl_info.setText(f"已保存: {Path(save_path).name}")
            QMessageBox.information(self, "保存成功", f"图片已保存至:\n{save_path}")

    # ==================== 缩略图更新 ====================
    def update_thumbs(self):
        self.thumb_list.clear()
        if self.original_img is None:
            return

        width = self.thumb_list.width()
        if width <= 0:
            width = 200

        item_width = int(width)
        item_width = max(80, min(item_width, width - 10))
        item_height = int(item_width * 0.7)

        for idx, bbox in enumerate(self.candidates[:self.top_n]):
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            crop = self.original_img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            thumb = cv2.resize(crop, (item_width, item_height))
            thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            h, w, ch = thumb_rgb.shape
            bytes_per_line = ch * w
            qt_img = QImage(thumb_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qt_img)

            label = QLabel()
            label.setPixmap(pix)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("background: transparent; border: none; padding: 0; margin: 0;")
            label.setFixedSize(item_width, item_height)

            item = QListWidgetItem()
            item.setData(Qt.UserRole, idx)
            item.setSizeHint(QSize(item_width, item_height))
            self.thumb_list.addItem(item)
            self.thumb_list.setItemWidget(item, label)

        self.thumb_list.setSpacing(6)

    def highlight_thumb(self, idx):
        for i in range(self.thumb_list.count()):
            item = self.thumb_list.item(i)
            if item.data(Qt.UserRole) == idx:
                self.thumb_list.setCurrentItem(item)
                return

    def on_thumb_clicked(self, item):
        idx = item.data(Qt.UserRole)
        if idx is not None and idx < len(self.candidates):
            self.current_idx = idx
            if self.current_mode == self.MODE_PREVIEW:
                bbox = self.candidates[idx]
                x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
                self.cropped_img = self.original_img[y1:y2, x1:x2]
                self.reset_params(silent=True)
            self.update_display()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            if self.current_mode == self.MODE_COMPARE:
                self.current_mode = self.MODE_PREVIEW
                self.btn_compare.setChecked(False)
            else:
                self.current_mode = self.MODE_COMPARE
                self.btn_compare.setChecked(True)
            self.update_display()
        else:
            super().keyPressEvent(event)