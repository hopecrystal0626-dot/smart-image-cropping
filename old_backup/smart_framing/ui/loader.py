# -*- coding: utf-8 -*-
"""
LoadWorker 后台加载线程 - 懒加载 process_image，避免顶层导入 torch
"""

import os
import sys
import traceback
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

# ======================================================
# 辅助函数：修复 DLL 路径（在子线程中也执行）
# ======================================================
def add_torch_lib_to_path():
    """将 torch 的 lib 目录添加到 DLL 搜索路径"""
    import site
    possible_paths = [
        os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib'),
        os.path.join(sys.prefix, 'lib', 'site-packages', 'torch', 'lib'),
    ]
    for site_dir in site.getsitepackages():
        alt_path = os.path.join(site_dir, 'torch', 'lib')
        if alt_path not in possible_paths:
            possible_paths.append(alt_path)
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            try:
                os.add_dll_directory(path)
                print(f"[子线程] 已添加 DLL 目录: {path}")
            except (AttributeError, OSError):
                os.environ['PATH'] = path + os.pathsep + os.environ.get('PATH', '')
                print(f"[子线程] 已通过 PATH 添加: {path}")
            return True
    print("[子线程] 警告: 未找到 torch/lib 目录")
    return False


def _safe_imread(img_path):
    """用 numpy 绕开 OpenCV 不支持中文路径的问题"""
    stream = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
    return img


# ======================================================
# LoadWorker 定义
# ======================================================
class LoadWorker(QThread):
    progress = pyqtSignal(int, str)
    # candidates_list, img_bgr, grid_bgr, depth_bgr, saliency_bgr
    finished = pyqtSignal(object, object, object, object, object)
    error = pyqtSignal(str)

    def __init__(self, img_path):
        super().__init__()
        self.img_path = img_path

    def run(self):
        try:
            # 1. 修复 DLL 路径（在子线程中）
            add_torch_lib_to_path()

            # 2. 修复 Python 路径（确保能找到 smart_framing）
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
                print(f"[子线程] 已将项目根目录加入 sys.path: {project_root}")

            # 3. 延迟导入 process_image（此时 torch 已被加载）
            from smart_framing.core.pipeline import process_image
            print("[子线程] 成功导入 process_image")

            self.progress.emit(5, "初始化...")

            # 4. 调用 process_image
            #    save_vis=False 只跳过写文件，grid_image 依然会生成（pipeline已修复）
            result = process_image(str(self.img_path), use_depth=True, save_vis=False)
            self.progress.emit(50, "特征提取与评分...")

            # 5. 用 numpy 安全读取原图（避免中文路径问题）
            img_bgr = _safe_imread(str(self.img_path))
            if img_bgr is None:
                raise ValueError(f"无法读取原图: {self.img_path}")

            # 6. 解析结果 —— 字段名与 pipeline.py 完全对齐
            #    pipeline 返回:
            #      best_box      : (x1,y1,x2,y2) 整数元组
            #      top10         : list of dict，含 box(元组)/score/aes/content/thirds/center
            #      top10_crops   : list of RGB ndarray
            #      grid_image    : RGB ndarray 或 None
            #      depth_map     : float ndarray 或 None
            #      saliency_mask : float 0~1 ndarray 或 None
            #      seg_map       : panoptic 分割图 或 None
            top10           = result.get('top10', [])
            top10_crops_rgb = result.get('top10_crops', [])
            grid_image_rgb  = result.get('grid_image', None)
            depth_map       = result.get('depth_map', None)
            saliency_mask   = result.get('saliency_mask', None)
            seg_map         = result.get('seg_map', None)

            self.progress.emit(70, "构建候选列表...")

            # 7. 构建 candidates_list（top10 与 top10_crops 一一对应）
            candidates_list = []
            pairs = list(zip(top10, top10_crops_rgb))
            if not pairs:
                # 兜底：至少用 best_box 构造一个候选
                best_box = result.get('best_box')
                best_crop_rgb = result.get('best_crop')
                if best_box is not None and best_crop_rgb is not None:
                    x1, y1, x2, y2 = best_box
                    crop_bgr = cv2.cvtColor(best_crop_rgb, cv2.COLOR_RGB2BGR)
                    candidates_list.append({
                        'x1': int(x1), 'y1': int(y1),
                        'x2': int(x2), 'y2': int(y2),
                        'final_score': 1.0,
                        'aes': 0, 'content': 0, 'thirds': 0, 'center': 0,
                        'crop_image': crop_bgr,
                    })
            else:
                for i, (info, crop_rgb) in enumerate(pairs):
                    # info['box'] 是 (x1,y1,x2,y2) 整数元组，直接解包
                    x1, y1, x2, y2 = info['box']
                    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                    candidates_list.append({
                        'x1': int(x1), 'y1': int(y1),
                        'x2': int(x2), 'y2': int(y2),
                        'final_score': float(info.get('score', 1.0 - i * 0.05)),
                        'aes':     float(info.get('aes', 0)),
                        'content': float(info.get('content', 0)),
                        'thirds':  float(info.get('thirds', 0)),
                        'center':  float(info.get('center', 0)),
                        'crop_image': crop_bgr,
                    })

            self.progress.emit(85, "生成可视化图像...")

            # 8. grid_image RGB → BGR
            grid_bgr = None
            if grid_image_rgb is not None:
                grid_bgr = cv2.cvtColor(grid_image_rgb, cv2.COLOR_RGB2BGR)

            # 9. 深度图：归一化后上色（INFERNO 伪彩色，直觉上近=亮）
            depth_bgr = None
            if depth_map is not None:
                d = depth_map.astype(np.float32)
                d_norm = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)
                depth_colored = cv2.applyColorMap(d_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
                depth_bgr = depth_colored  # BGR，可直接传给 set_image

            # 10. 语义分割图：seg_map 是整数标签图，转为伪彩色
            saliency_bgr = None
            if seg_map is not None:
                # seg_map 可能是 H×W int32，映射为 BGR 伪彩色
                seg = seg_map.astype(np.float32)
                seg_norm = cv2.normalize(seg, None, 0, 255, cv2.NORM_MINMAX)
                saliency_bgr = cv2.applyColorMap(seg_norm.astype(np.uint8), cv2.COLORMAP_TURBO)
            elif saliency_mask is not None:
                # 备用：如果 seg_map 没有，用显著性图代替
                s_norm = (saliency_mask * 255).astype(np.uint8)
                saliency_bgr = cv2.applyColorMap(s_norm, cv2.COLORMAP_JET)

            self.progress.emit(95, "处理完成")
            self.finished.emit(candidates_list, img_bgr, grid_bgr, depth_bgr, saliency_bgr)

        except Exception as e:
            traceback.print_exc()
            self.error.emit(traceback.format_exc())