# -*- coding: utf-8 -*-
"""
WorkflowPanel 工作流进度面板
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar
from PyQt5.QtCore import Qt


class WorkflowPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.step_icons = ["🔍", "🏷️", "📦", "⭐", "🎯"]
        self.step_names = ["显著性检测", "语义分割", "候选框生成", "智能评分", "最优选择"]
        self.current_step = -1
        self.step_labels = []
        self.progress_bar = None
        self.status_label = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        title = QLabel("🤖 AI取景工作流")
        title.setStyleSheet("font-weight: bold; color: #1f2937;")
        layout.addWidget(title)

        steps_widget = QWidget()
        steps_layout = QHBoxLayout(steps_widget)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        steps_layout.setSpacing(8)

        self.step_labels = []
        for i, (icon, name) in enumerate(zip(self.step_icons, self.step_names)):
            label = QLabel(f"{icon} {name}")
            label.setStyleSheet("""
                padding: 10px 12px;
                border-radius: 4px;
                background: #f3f4f6;
                color: #9ca3af;
            """)
            label.setAlignment(Qt.AlignCenter)
            label.setWordWrap(True)
            self.step_labels.append(label)
            steps_layout.addWidget(label)

        layout.addWidget(steps_widget)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                height: 6px;
                border: none;
                border-radius: 3px;
                background: #e5e7eb;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4f46e5, stop:1 #10b981);
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("等待开始...")
        self.status_label.setStyleSheet("color: #6b7280;")
        layout.addWidget(self.status_label)

    def update_step(self, step, message="", progress=0):
        self.current_step = step
        if step >= len(self.step_labels):
            for label in self.step_labels:
                label.setStyleSheet("""
                    padding: 10px 12px;
                    border-radius: 4px;
                    background: #d1fae5;
                    color: #065f46;
                """)
        else:
            for i, label in enumerate(self.step_labels):
                if i < step:
                    label.setStyleSheet("""
                        padding: 10px 12px;
                        border-radius: 4px;
                        background: #d1fae5;
                        color: #065f46;
                    """)
                elif i == step:
                    label.setStyleSheet("""
                        padding: 10px 12px;
                        border-radius: 4px;
                        background: #e0e7ff;
                        color: #3730a3;
                        font-weight: bold;
                    """)
                else:
                    label.setStyleSheet("""
                        padding: 10px 12px;
                        border-radius: 4px;
                        background: #f3f4f6;
                        color: #9ca3af;
                    """)
        if self.progress_bar:
            self.progress_bar.setValue(progress)
        if self.status_label:
            self.status_label.setText(message)

    def reset(self):
        self.current_step = -1
        for label in self.step_labels:
            label.setStyleSheet("""
                padding: 10px 12px;
                border-radius: 4px;
                background: #f3f4f6;
                color: #9ca3af;
            """)
        self.progress_bar.setValue(0)
        self.status_label.setText("等待开始...")