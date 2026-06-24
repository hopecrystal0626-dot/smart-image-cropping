import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- 数据与输出 ----------
DATA_DIR = os.path.join(BASE_DIR, "data", "testD")       # 仅用于原批处理脚本，pipeline 不依赖
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output", "final_ranker")

# ---------- 权重路径 ----------
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
U2NET_MODEL_PATH = os.path.join(WEIGHTS_DIR, "u2net.pth")
AESTHETIC_MODEL_PATH = os.path.join(WEIGHTS_DIR, "ava+logos-l14-linearMSE.pth")
YOLO_MODEL_PATH = os.path.join(WEIGHTS_DIR, "yolov8n-seg.pt")   # 若不存在，ranker 会跳过

# ---------- 模型名称（HuggingFace） ----------
DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
PANOPTIC_MODEL_NAME = "facebook/mask2former-swin-base-coco-panoptic"

# ---------- 初筛参数 ----------
NMS_IOU_THRESH = 0.85
SUBJECT_COVERAGE_THRESH = 0.6
KEEP_TOP_N = 100

# ---------- 融合评分权重 ----------
W_AES     = 0.29
W_CONTENT = 0.20
W_THIRDS  = 0.22
W_CENTER  = 0.30
W_DEPTH_PENALTY   = 0.25
W_CLIP_PENALTY    = 0.60
W_MISSING_PENALTY = 0.50
W_YOLO_PENALTY    = 0.00          # 若启用 YOLO 可调大

# ---------- 各评分子参数 ----------
CONTENT_STD_SATURATE = 60.0
CONTENT_EDGE_SATURATE = 15.0
THIRDS_POSITIONS = [1/3, 2/3]
CLIP_COVERAGE_TOLERANCE = 0.97
LARGE_OBJECT_THRESHOLD = 0.45
DEPTH_STD_SATURATE = 0.25

# ---------- 截断惩罚参数（ranker 中用到的） ----------
CLIP_PENALTY_WEIGHT = 0.5      # 单个实例最大惩罚值
CLIP_FULL_PENALTY_AT = 0.4     # 切掉40%时达到最大惩罚

# ---------- 人物/动物标签 ----------
PERSON_LIKE_LABELS = {"person", "dog", "cat", "horse", "bird", "cow",
                      "sheep", "bear", "elephant", "zebra", "giraffe"}