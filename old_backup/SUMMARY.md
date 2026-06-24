项目文件说明汇总 — smart-image-cropping
=====================================

一、整体工程逻辑（简述）

- 输入：原始图片（data/testA 下的样本）
- 候选框生成：`crop/candidate_generator.py` 生成多尺度滑动窗口候选框
- 显著性筛选：`saliency/detector.py`（默认 FT）生成显著图，`saliency/saliency_utils.py` 根据显著性与中心偏好筛选候选框
- 完整性检测：使用 YOLO（和 MTCNN 兜底）检测场景中的人体/物体，剔除会截断重要物体的候选框
- 美学/构图评分：多种实现（`composition/` 和 `aesthetic/`）计算三分法、平衡、留白、主体保留等子评分，组合成综合得分；`composition/aesthetic_scorer.py` 封装了 NIMA（pyiqa）用于美学评分
- CLIP 重排序：`clip_score/` 提供基于 CLIP 的文本提示式美学评分，可对候选框做平滑重排序或直接排名（用于方案三）
- 输出：按策略输出 top-N 候选框、保存可视化与最佳裁剪（各种 scripts 在 `experiments/` 下提供演示/批量测试）

二、主要文件与简短说明（按模块）

- crop_system_great.py：主控脚本（加权融合手工分+NIMA，后接 CLIP 平滑重排序）、批量处理与可视化。
- crop_system2.py：与上类似的另一个实现/副本。
- rank_by_clip.py：使用 CLIP 对显著性筛选后候选框进行评分与可视化，支持多模式切换。
- main.py：占位（空）。

- aesthetic/：一组基于显著图的局部评分函数（`thirds.py`, `balance.py`, `whitespace.py`, `subject_preservation.py`, `object_preservation.py`, `composition_score.py`），用于计算构图细项并给出组合分。

- composition/：更完整的构图与流水线实现，包括 `CompositionScorer`（细粒度分项与权重），`AestheticPipeline`（整合显著性、人体/物体检测、NIMA、美学与保底策略）、以及 MTCNN/YOLO 封装的检测器。

- crop/：`BBox` 数据类与工具、候选框生成器、以及简单裁剪函数。

- saliency/：显著性检测器实现（默认 FT）与工具函数（显著图获取、后处理、候选框按分位筛选、主体中心/主体 bbox 提取）。

- clip_score/：CLIP 模型封装（transformers）及基于 prompt 的美学评分实现，支持批量评分与多模式（balanced/portrait/landscape/...）。

- experiments/：若干演示与批量测试脚本（`detect.py`, `batch_fusion.py`, `handcraft_system.py`, `test_saliency.py`, `visualize_*` 等），用于调试、可视化与统计评估。

- evaluation/、utils/、gui/、explanation/：部分为占位或未实现文件（空文件或未完成的模块）。

三、注意事项与建议

- 仓库中存在多个相似/重复实现（例如 `aesthetic/` 与 `composition/` 中都有三分法、平衡、留白实现；`crop_system*` 有多个版本）。建议在需要生产化前确定一套主线实现并去重。 
- 运行依赖：OpenCV、ultralytics YOLO、transformers、pyiqa、torch 等；部分模块会在线下载模型或依赖（注意网络/模型路径配置）。

如果你希望我把这份汇总写入其它格式（如 `SUMMARY.md` 的更详细版、或生成模块依赖图），或要我把重复实现差异列出来，请告诉我。

四、发现的问题 / 风险 与 优化建议（详尽版）

1) 总体流程回顾

- 输入图片 → 候选框生成（滑动窗口、多尺度）→ 显著性筛选 → 完整性检测（YOLO + MTCNN 兜底）
- → 美学/构图评分（handcraft / CompositionScorer / NIMA）→（可选）CLIP 精排或平滑重排序 → 输出 Top-N、可视化、保存

2) 主要问题与风险（需要优先注意）

- 模块重复与实现不一致：`aesthetic/` 与 `composition/`、多个 `crop_system*` 导致维护困难和行为不确定性。
- 硬编码路径与模型缓存：如 `D:/AI_Models`、MTCNN_HOME、yolov8 模型路径等，跨环境不可移植。
- 不稳健的模型下载逻辑：自动下载失败只打印错误，缺重试与完整性校验。
- 分散的超参与阈值：权重、阈值、比例分散在各文件，难以统一调参与复现实验。
- 单例/全局对象无并发保护：CLIP、pyiqa 等单例在并发情形下可能重复加载或竞态。
- 内存/显存未保护：对 CLIP/NIMA/YOLO 的批量调用没有显存/批次限制，存在 OOM 风险。
- 输入校验不一致：对灰度图、空裁剪、越界 bbox 的处理并不统一，可能导致异常。
- 日志与错误处理薄弱：大量 `print`，缺统一日志框架和错误上报/重试策略。
- 安全隐患：直接从外部镜像下载模型未校验完整性或签名。
- 接口不统一：不同函数返回值格式不一致（`BBox` vs tuple，dict 字段不同），易出错。

3) 功能/逻辑层面的潜在漏洞

- 裁剪区域可能为空或越界，传入评分器（NIMA/CLIP）会报错或返回异常值；需统一裁剪前校验。
- 完整性阈值（如 0.85）可能对某些场景过强，需在验证集上调整或作为可配置参数。
- CLIP 平滑重排序的阈值/压缩逻辑为经验数值，未在多数据集上验证，可能破坏排序一致性。
- IoU 与 bbox 表示混用（有时用 `BBox`，有时用 tuple），会引发类型错误或错误计算。

4) 性能与算法优化建议（按优先级）

- 高优先（即刻可做）
	- 集中配置：在 `utils/config.py` 定义统一的 `MODELS_DIR`, `DEVICE`, `YOLO_PATH`, `CLIP_BATCH_SIZE`, `DEFAULT_THRESHOLDS` 等。
	- 限制批次大小：CLIP/NIMA 批量评分前分小批（例如 <=16），并在 OOM 时自动降级或回退 CPU。
	- 统一输入校验：实现 `safe_read_image()` / `safe_crop()`，确保传入评分器的图像非空且通道符合预期。
	- 统一日志：实现 `utils/logger.py` 并替换关键处的 `print`，方便排查与统计。

- 中期（效果显著，需一定开发量）
	- 候选框两阶段筛选：先用轻量指标（显著性平均、中心距离、面积范围）做粗筛，再对前 K 做 NIMA/CLIP 精排。
	- 去冗余：对候选做 NMS/聚类，减少高度重叠候选，降低后续评分次数。
	- 统一评分接口：定义 `Scorer` 抽象类（score_single/score_batch），将 NIMA/CLIP/handcraft 适配进去，便于融合与替换。
	- 参数学习：在验证集上用线性模型学习手工分与 NIMA/CLIP 的融合权重，取代固定经验权重。

- 长期（高级优化）
	- 训练轻量预筛模型：用小型 CNN/MLP 训练好/不好分类器，作为 NIMA/CLIP 的前置筛选器。
	- 特征级复用：批量提取候选图像特征并缓存（CLIP image encoder），在不同评分器间复用。
	- 半精度推理：在支持平台上启用 FP16 推理以节省显存和提高吞吐。

5) 工程和可维护性建议

- 添加依赖文件：`requirements.txt` 或 `environment.yml` 明确版本，便于复现。
- 统一模型目录与下载校验：指定 `MODELS_DIR` 并添加下载后哈希校验与重试机制。
- 合并重复模块：选择一套主实现（建议保留 `composition/CompositionScorer` 与 `composition/AestheticPipeline` 作为主线），把 `aesthetic/` 做为工具库或迁移并删除冗余代码。
- 增加 smoke tests：脚本对 1~5 张图片跑完整 pipeline，记录耗时/内存/输出，方便回归测试。
- 编写小规模基准：统计 NIMA/CLIP 单次耗时、平均分布、CLIP 分差分布，用数据决策阈值。

6) 小步实施计划（建议次序）

- 第 1 步（1–3 天）: 配置 + 日志 + 批次保护
	1. 在 `utils/config.py` 定义基础配置并在关键模块引用。
	2. 在 `utils/logger.py` 配置 logging，并替换主要 print。
	3. 在 CLIP/NIMA 调用处强制 `max_batch_size` 并在 OOM 时降级。

- 第 2 步（1–2 周）: 候选两阶段筛与统一评分接口
	1. 实现候选粗筛（显著性/中心/面积）→ NMS → 精排流程。
	2. 统一 `Scorer` 接口并用配置控制采用哪种 scorer。
	3. 在小验证集上网格搜索 alpha/beta/clip 权重或用线性回归学习融合权重。

- 第 3 步（长期）: 训练预筛模型、特征缓存与自动化
	1. 训练轻量预筛二分类器以节省昂贵评分。
	2. 实现特征缓存（CLIP image encoder 特征）并在多个脚本间复用。
	3. 加入 CI、基准和自动超参搜索。

七、我可以做的下一步（选择一项）

- A. 生成 `utils/config.py` 和 `utils/logger.py` 的建议草案（不改现有逻辑，只生成文件内容供你审阅）。
- B. 列出 `aesthetic/` 与 `composition/` 两套实现的逐项差异对照表，帮助你决定保留或合并哪一套。
- C. 帮你写一个 smoke-test 脚本（单线程、单图/多图跑通并记录耗时与内存）以量化当前开销。

请告诉我你想先做哪一项（A/B/C），我就开始准备相应的文件或对比表。
