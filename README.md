# smart-image-cropping

## 运行入口

主入口：

```powershell
python main.py path\to\image.jpg --top_k 10 --clip_mode balanced
```

输出结果包含：
- 最佳候选框坐标
- 最佳裁剪图
- 原图上带最佳框的可视化图

## 测试入口

按步骤测试：

```powershell
python experiments/steps/test_candidates.py path\to\image.jpg
python experiments/steps/test_saliency_step.py path\to\image.jpg --top_percent 0.3
python experiments/steps/test_fusion_step.py path\to\image.jpg
python experiments/steps/test_end_to_end.py path\to\image.jpg
```

## 当前结构

- `crop/`：候选框与裁剪工具
- `saliency/`：显著性检测与筛选
- `composition/`：构图、美学、主体保留相关
- `clip_score/`：CLIP 评分
- `pipeline/`：主流程配置与执行器
- `experiments/steps/`：分步测试脚本