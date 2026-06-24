#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
智能取景系统 GUI - 入口文件（备选）
设置环境变量并启动 GUI，兼容任意运行位置
"""

import os
import sys
from pathlib import Path
import torch

# 将项目根目录（smart-image-cropping）添加到 sys.path
project_root = Path(__file__).parent.parent 
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


from smart_framing.ui.gui import main

if __name__ == "__main__":
    main()