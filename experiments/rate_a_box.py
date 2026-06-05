"""
集成候选框生成 + 构图评分 + Top-K可视化
"""

import cv2
import sys
import os

# ========== 项目路径 ==========
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from composition.composition_score import CompositionScorer
from crop.candidate_generator import generate_candidates


# =========================
# 工具：BBox转换
# =========================
def box_to_bbox(box):
    x = box.x1
    y = box.y1
    w = box.x2 - box.x1
    h = box.y2 - box.y1
    return (x, y, w, h)


# =========================
# Top-K 可视化
# =========================
def visualize_top_k(img, results, k=15):
    canvas = img.copy()

    for i in range(min(k, len(results))):
        x, y, w, h = results[i]["bbox"]
        score = results[i]["score"]

        # 排名颜色（越靠前越绿）
        color = (0, 255 - i * 10, i * 10)

        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)

        cv2.putText(
            canvas,
            f"{i+1}:{score:.3f}",
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    cv2.imshow("Top-K Candidates", cv2.resize(canvas, (900, 700)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =========================
# 主函数
# =========================
def main():

    # ========== 1. 读取图片 ==========
    img_path = r"data/testA/A08.jpg"
    img = cv2.imread(img_path)

    if img is None:
        print("❌ 图片读取失败")
        return

    h, w = img.shape[:2]
    print(f"📷 图片尺寸: {w} x {h}")
    print("=" * 50)

    # ========== 2. 生成候选框 ==========
    print("🔍 生成候选框...")
    boxes = generate_candidates(img_w=w, img_h=h)
    print(f"✅ 候选框数量: {len(boxes)}")

    # 👉 可选：快速测试（避免600太慢）
    # boxes = boxes[:200]

    # ========== 3. 构图评分 ==========
    print("\n🎯 开始评分...")

    scorer = CompositionScorer()   # ⭐ 只初始化一次

    results = []

    for i, box in enumerate(boxes):

        bbox = box_to_bbox(box)

        # ⭐ 单样本评分（最快正确方式）
        detail = scorer.compute_single_score(img, bbox)

        results.append({
            "bbox": bbox,
            "score": detail["total_score"],
            "detail": detail
        })

        # 进度打印
        if i % 50 == 0:
            print(f"  处理进度: {i}/{len(boxes)}")

    print("✔ 评分完成")

    # ========== 4. 排序 ==========
    results.sort(key=lambda x: x["score"], reverse=True)

    best = results[0]

    print("\n" + "=" * 50)
    print("🏆 最佳候选框")
    print("=" * 50)
    print(f"bbox: {best['bbox']}")
    print(f"score: {best['score']:.4f}")
    print("\n📊 详细得分:")
    print(best["detail"])

    # ========== 5. Top-K 可视化 ==========
    visualize_top_k(img, results, k=15)

    # ========== 6. 单独最佳框显示 ==========
    show = input("\n是否显示最佳框(y/n): ")

    if show.lower() == 'y':

        canvas = img.copy()
        x, y, w_box, h_box = best["bbox"]

        cv2.rectangle(canvas, (x, y), (x + w_box, y + h_box), (0, 255, 0), 3)

        cv2.putText(
            canvas,
            f"BEST: {best['score']:.4f}",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Best Result", cv2.resize(canvas, (800, 600)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()