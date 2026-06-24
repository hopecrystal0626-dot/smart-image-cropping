# 智能取景系统 - 运行说明

## 环境要求
- Python 3.9 ~ 3.11（推荐 3.10）
- Windows 10/11 64位
- CUDA 11.8+（可选，有 GPU 会更快；无 GPU 也可运行）

## 快速运行（源代码方式）

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 启动程序
```bash
python smart_framing/run.py
```

---

## 直接运行可执行文件

双击 `dist/智能取景/智能取景.exe` 即可。

> 注意：可执行文件已包含所有模型，无需安装 Python 或任何依赖，但**首次启动较慢**（约 30~60 秒），请耐心等待模型加载完成。

---