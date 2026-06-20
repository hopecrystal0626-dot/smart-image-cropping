#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
智能取景系统 GUI 入口（适配队友 process_image）
"""

import os
'''
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["OPEN_CLIP_CACHE"] = "D:/AI_Models/clip_cache"
os.environ["TORCH_HOME"] = "D:/AI_Models/torch_cache"
'''

import sys
# 添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from main_window import MainWindow
from PyQt5.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())