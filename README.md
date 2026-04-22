# MAR Bianyuan / MM5 Multimodal Calibration Benchmark

[中文说明](#中文说明) | [English](#english-version)

---

## 中文说明

### 1. 这个项目是做什么的

这是一个围绕 **MM5 多模态数据** 搭建的实验工作区，核心目标是把不同的跨模态标定 / 配准思路放到同一个 benchmark 里做可复现对比，并把作者已有的 **MAR 历史流程** 与新的基线方法统一整理到一个仓库中。

如果你第一次接触这个项目，可以先把它理解成三部分：

1. `mm5_calib_benchmark/`
   这里是主代码。它负责读取 MM5 数据索引、运行不同标定 / 对齐方法、计算指标、导出可视化结果和对比表。
2. `mar_scholar_compare/`
   这里是面向 Scene 282 的文献对比与边界分析材料，更偏“说明”和“对比展示”。
3. `runs/`
   这里存放阶段性图像、文档和一些人工分析结果，属于实验记录区。

简单说，这个仓库不是单一算法实现，而是一个 **多方法对比平台 + 历史 MAR 结果复现实验区 + Scene 282 专项分析区**。

### 2. 项目想回答什么问题

这个项目重点关注下面几个问题：

1. 在 MM5 数据上，传统平面单应、棋盘格 stereo、深度辅助配准、MAR 风格边界细化、自校准方法，谁更稳？
2. 不同方法在 `thermal` 和 `uv` 两类目标模态上的表现差异是什么？
3. 历史 MAR 流程在 Scene 282 上的“论文版 / 工程版”结果，与新 benchmark 中的各类方法相比处于什么位置？
4. 单看 `pixel_accuracy` 容易高估效果，是否需要联合 `mean_iou`、`boundary_f1`、`keypoint_transfer_error_px` 一起判断？

### 3. 仓库结构

```text
MAR_bianyuan/
|- calibration/
|  |- 标定文件、相机参数、棋盘格相关配置
|- mar_scholar_compare/
|  |- Scene 282 相关文献对比材料
|  |- 边界分析与辅助边缘检测结果
|- mm5_calib_benchmark/
|  |- configs/                # 全局配置与各方法配置
|  |- methods/                # M0~M7 各方法实现
|  |- eval/                   # 指标计算
|  |- viz/                    # 可视化叠图、热力图等
|  |- scripts/                # 入口脚本
|  |- outputs/mm5_benchmark/  # 已生成的 benchmark 结果
|- runs/
|  |- Word 文档、图像、人工分析与对比输出
|- .gitignore
`- README.md
```

### 4. 核心代码入口

最重要的几个入口脚本如下：

1. `mm5_calib_benchmark/scripts/run_all_methods.py`
   运行默认方法集合，并生成 Scene 282-3 的综合对比图。
2. `mm5_calib_benchmark/scripts/make_splits.py`
   根据索引 CSV 生成 `train / val / test` 划分。
3. `mm5_calib_benchmark/scripts/make_scene_2823_comparison.py`
   只重新生成 Scene 282-3 的对比图和汇总材料。
4. `mm5_calib_benchmark/scripts/run_legacy_mar_scene282.py`
   调用外部同级目录中的 `MAR_test/backup_2600.py`，复现实验室历史 MAR 全流程结果。
5. `mm5_calib_benchmark/scripts/generate_algorithm_summary_doc.py`
   基于现有输出结果生成 Word 总结文档。

主流程代码在 [mm5_calib_benchmark/pipeline.py](/e:/aa_read_yan/aMAR/MAR_bianyuan/mm5_calib_benchmark/pipeline.py:1)，配置加载逻辑在 [mm5_calib_benchmark/config.py](/e:/aa_read_yan/aMAR/MAR_bianyuan/mm5_calib_benchmark/config.py:1)。

### 5. 当前已经纳入 benchmark 的方法

这个仓库目前实现了 `M0 ~ M7` 八类方法：

1. `M0 MM5 Official`
   直接使用 MM5 官方 stereo / 标定结果，再做平面投影和场景级微调。
2. `M1 Zhang`
   使用棋盘格观测进行 OpenCV / Zhang 风格 stereo 标定，再做平面对齐。
3. `M2 Su2025 XoFTR Fallback`
   模拟跨模态特征匹配思路，但当前工程实现采用的是 fallback 特征匹配，而不是完整 transformer 权重版。
4. `M3 Jay2025 SGM`
   面向 UV 方向的一类对比方法。
5. `M4 Muhovic DepthBridge`
   使用深度图辅助跨模态对齐。
6. `M5 EPnP Baseline`
   使用深度相关几何信息建立较强基线。
7. `M6 MAR Edge Refine`
   在已有对齐基础上做 MAR 风格边界细化。
8. `M7 Depth Guided Self Calibration`
   当前仓库里最完整、也最偏作者方案的一条路线，结合全局姿态优化、深度引导和边界细化。

从现有 `scene_282_3_metric_explanation.md` 的结果摘要看，当前 test 集上 `M7` 是最强方法之一，`M5`、`M4`、`M6` 处于中间梯队。

### 6. 输入数据是怎样组织的

这个项目不是“纯代码仓库”，它依赖 MM5 的原始数据、对齐数据、标定目录和标注文件。

当前索引文件位于：

- [mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv](/e:/aa_read_yan/aMAR/MAR_bianyuan/mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv:1)

从这个 CSV 的表头可以看出，benchmark 会使用这些信息：

1. 原始 RGB、thermal、UV、IR、depth 图像路径
2. 对齐后的参考图像路径
3. RGB / thermal / UV 的语义标注路径
4. 元信息 JSON
5. 标定根目录 `calibration_root`
6. 每个样本是否具备某种模态，以及数据划分 `split`

当前仓库里保存的是索引和结果，不是完整 MM5 原始数据拷贝。也就是说：

1. 你可以直接查看已生成结果
2. 但如果要重新跑 benchmark，CSV 中指向的原始数据路径需要在本机真实存在
3. 当前索引里大量路径指向 `D:\a三模数据\...`

### 7. 运行依赖

从代码导入可以看出，当前项目主要依赖这些 Python 包：

1. `numpy`
2. `opencv-python`
3. `scipy`
4. `scikit-image`
5. `python-docx`

Python 版本建议与当前环境保持一致，当前仓库使用的是 `.venv`，并已在本机上通过 `python 3.13.3` 运行过部分脚本。

### 8. 最常用的运行方式

在仓库根目录执行。

#### 8.1 生成 train / val / test 划分

```powershell
.venv\Scripts\python.exe -m mm5_calib_benchmark.scripts.make_splits
```

#### 8.2 运行默认 benchmark 套件

```powershell
.venv\Scripts\python.exe -m mm5_calib_benchmark.scripts.run_all_methods
```

它会做两件事：

1. 运行默认方法集合 `DEFAULT_SUITE`
2. 生成 Scene 282-3 的综合对比图

#### 8.3 仅生成 Scene 282-3 汇总图

```powershell
.venv\Scripts\python.exe -m mm5_calib_benchmark.scripts.make_scene_2823_comparison
```

#### 8.4 复现实验室旧版 MAR Scene 282 流程

```powershell
.venv\Scripts\python.exe -m mm5_calib_benchmark.scripts.run_legacy_mar_scene282 --mar-mode all --save-level debug
```

注意：这个脚本不是完全自包含的。它要求在当前仓库同级目录存在一个 `MAR_test` 目录，并且里面有 `backup_2600.py` 和相关资源。

也就是说，默认期望目录关系类似这样：

```text
.../aMAR/
|- MAR_bianyuan/
`- MAR_test/
```

### 9. 输出结果会保存到哪里

主输出目录是：

- [mm5_calib_benchmark/outputs/mm5_benchmark](/e:/aa_read_yan/aMAR/MAR_bianyuan/mm5_calib_benchmark/outputs/mm5_benchmark:1)

其中比较重要的子目录有：

1. `method_M*_.../`
   每个方法、每个模态方向各自的输出目录，里面通常包括：
   `calib/`、`metrics/`、`masks/`、`viz/`、`warped/`
2. `legacy_mar_scene282_reproduced/`
   旧 MAR 流程在 Scene 282 上的复现实验结果
3. `scene_282_3_comparison/`
   汇总对比图、排名表、指标说明、历史说明
4. `splits/`
   数据划分结果和带 split 字段的索引 CSV

### 10. 应该如何阅读结果

如果你是第一次看这个项目，建议按这个顺序读：

1. 先看根目录 `README`
2. 再看 [mar_scholar_compare/README.md](/e:/aa_read_yan/aMAR/MAR_bianyuan/mar_scholar_compare/README.md:1)
3. 再看 `mm5_calib_benchmark/outputs/mm5_benchmark/scene_282_3_comparison/`
4. 重点看下面两个说明文件：
   [scene_282_3_metric_explanation.md](/e:/aa_read_yan/aMAR/MAR_bianyuan/mm5_calib_benchmark/outputs/mm5_benchmark/scene_282_3_comparison/scene_282_3_metric_explanation.md:1)
   [scene_282_3_mar_history_note.md](/e:/aa_read_yan/aMAR/MAR_bianyuan/mm5_calib_benchmark/outputs/mm5_benchmark/scene_282_3_comparison/scene_282_3_mar_history_note.md:1)

几个关键指标的直观含义：

1. `pixel_accuracy`
   像素分类整体正确率。数值可能很高，但容易被大面积背景“抬高”。
2. `mean_iou`
   更敏感地反映前景区域和类别区域是否真的对齐。
3. `boundary_f1`
   更关注目标边界是否贴合。
4. `keypoint_transfer_error_px`
   更接近“几何偏差到底有多少像素”的指标。

所以这个项目不建议只看一个指标，而是要把语义、边界和几何三类指标一起看。

### 11. 当前仓库中已经包含了什么

这个仓库当前已经提交了大量实验产物，因此它既是代码仓库，也是结果仓库。当前你可以直接在仓库中看到：

1. 方法实现代码
2. 标定 YAML 和若干 calibration 文件
3. 已生成的 benchmark 结果
4. Scene 282 相关图像、对比图和 Word 文档
5. 历史 MAR 流程复现实验输出

这对“快速查看结果”很方便，但也意味着仓库体积会比较大。

### 12. 项目当前的限制

这是理解这个仓库时最重要的几条限制：

1. 仓库并不包含完整 MM5 原始数据，只包含索引和已生成结果
2. 某些方法名称对应的是“工程近似实现”，不一定是原论文的完整官方复现
3. `run_legacy_mar_scene282.py` 依赖外部同级目录 `MAR_test`
4. 当前很多输出是为作者当前机器路径组织的，换机器跑时往往需要先修正索引路径
5. 仓库里同时存在代码、结果、图像和文档，因此新协作者第一次看时容易混淆“源码”和“实验产物”

### 13. 如果你是新接手这个项目的人，建议从哪里开始

建议按照下面顺序上手：

1. 看这份 `README`
2. 理解 `mm5_calib_benchmark/` 是主代码，`runs/` 更像实验记录，`mar_scholar_compare/` 更像分析展示
3. 先阅读 [mm5_calib_benchmark/pipeline.py](/e:/aa_read_yan/aMAR/MAR_bianyuan/mm5_calib_benchmark/pipeline.py:1)，了解总流程
4. 再看 `methods/` 下每个 `run.py`
5. 最后根据自己机器上的数据路径，决定是“只复查现有结果”，还是“重新跑 benchmark”

---

## English Version

### 1. What this repository is

This repository is a **working research benchmark for MM5 multimodal calibration and registration**. Its main purpose is to place multiple cross-modal calibration / alignment strategies into one reproducible pipeline, while also preserving the author's historical **MAR workflow** and the scene-specific analysis around **Scene 282**.

If you are new to this project, the easiest mental model is:

1. `mm5_calib_benchmark/`
   The main codebase. It loads MM5 index files, runs multiple methods, evaluates metrics, and exports comparison figures.
2. `mar_scholar_compare/`
   Supplemental analysis and literature-oriented comparison material, especially for Scene 282.
3. `runs/`
   Experiment records, generated figures, Word documents, and manual analysis artifacts.

So this is not a single-algorithm repository. It is a **benchmark workspace + historical MAR reproduction workspace + scene-focused analysis workspace**.

### 2. Main research questions

This project is designed to answer questions such as:

1. On MM5 data, how do planar homography, checkerboard stereo calibration, depth-assisted alignment, MAR-style edge refinement, and self-calibration compare under a unified benchmark?
2. How do methods behave differently on `thermal` versus `uv` target modalities?
3. Where do the historical MAR "paper" and "engineered" pipelines stand relative to the new benchmarked methods on Scene 282?
4. Why is `pixel_accuracy` alone insufficient, and why should it be interpreted together with `mean_iou`, `boundary_f1`, and `keypoint_transfer_error_px`?

### 3. Repository layout

```text
MAR_bianyuan/
|- calibration/
|  |- Calibration YAML files and camera-related assets
|- mar_scholar_compare/
|  |- Scene 282 comparison material and auxiliary edge analysis
|- mm5_calib_benchmark/
|  |- configs/                # Global and per-method config files
|  |- methods/                # Implementations for M0~M7
|  |- eval/                   # Metric computation
|  |- viz/                    # Overlay / heatmap visualization helpers
|  |- scripts/                # Main entry points
|  |- outputs/mm5_benchmark/  # Generated benchmark outputs
|- runs/
|  |- Reports, images, and manual analysis outputs
|- .gitignore
`- README.md
```

### 4. Main entry points

The most important scripts are:

1. `mm5_calib_benchmark/scripts/run_all_methods.py`
   Runs the default benchmark suite and generates the Scene 282-3 comparison summary.
2. `mm5_calib_benchmark/scripts/make_splits.py`
   Builds `train / val / test` splits from the index CSV.
3. `mm5_calib_benchmark/scripts/make_scene_2823_comparison.py`
   Regenerates only the Scene 282-3 summary artifacts.
4. `mm5_calib_benchmark/scripts/run_legacy_mar_scene282.py`
   Replays the historical MAR pipeline by calling `backup_2600.py` from an external sibling `MAR_test` repository.
5. `mm5_calib_benchmark/scripts/generate_algorithm_summary_doc.py`
   Produces a Word report from benchmark outputs.

The central orchestration logic lives in [mm5_calib_benchmark/pipeline.py](/e:/aa_read_yan/aMAR/MAR_bianyuan/mm5_calib_benchmark/pipeline.py:1), and configuration loading lives in [mm5_calib_benchmark/config.py](/e:/aa_read_yan/aMAR/MAR_bianyuan/mm5_calib_benchmark/config.py:1).

### 5. Implemented methods

The benchmark currently includes eight method families, `M0` through `M7`:

1. `M0 MM5 Official`
   Uses official MM5 stereo calibration as the baseline, followed by planar projection and scene tuning.
2. `M1 Zhang`
   Re-estimates stereo calibration from checkerboard observations using a Zhang/OpenCV-style workflow, then applies planar alignment.
3. `M2 Su2025 XoFTR Fallback`
   A practical approximation of a cross-modal feature matching idea; the current implementation uses fallback local matching rather than the original full transformer-based pipeline.
4. `M3 Jay2025 SGM`
   A comparison method focused on the UV branch.
5. `M4 Muhovic DepthBridge`
   Uses depth to guide alignment.
6. `M5 EPnP Baseline`
   A stronger geometry-aware baseline using depth-related cues.
7. `M6 MAR Edge Refine`
   Adds MAR-style boundary refinement after an initial alignment.
8. `M7 Depth Guided Self Calibration`
   The most complete "author-style" path in this repository, combining pose refinement, depth guidance, and boundary-aware improvement.

Based on the current benchmark summary, `M7` is one of the strongest methods on the current test subset, with `M5`, `M4`, and `M6` forming the next tier.

### 6. Input data expectations

This is not a self-contained toy repository. It depends on MM5 raw data, aligned reference data, annotations, calibration directories, and an index CSV describing all paths.

The active index file is:

- [mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv](/e:/aa_read_yan/aMAR/MAR_bianyuan/mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv:1)

From its header, the benchmark expects:

1. Raw RGB, thermal, UV, IR, and depth image paths
2. Aligned reference image paths
3. Semantic annotation paths for RGB / thermal / UV
4. Metadata JSON files
5. A calibration root path
6. Modality availability flags and the dataset split label

This repository already stores outputs and index files, but not the full underlying MM5 raw dataset. In practice, that means:

1. You can inspect generated results immediately
2. You cannot fully rerun the benchmark unless the CSV paths exist on your machine
3. Many current paths point to `D:\a三模数据\...`

### 7. Python dependencies

Based on the imports, the main Python dependencies are:

1. `numpy`
2. `opencv-python`
3. `scipy`
4. `scikit-image`
5. `python-docx`

The current local environment uses `.venv`, and Python `3.13.3` is available in this workspace.

### 8. Typical commands

Run all commands from the repository root.

#### 8.1 Build train/val/test splits

```powershell
.venv\Scripts\python.exe -m mm5_calib_benchmark.scripts.make_splits
```

#### 8.2 Run the default benchmark suite

```powershell
.venv\Scripts\python.exe -m mm5_calib_benchmark.scripts.run_all_methods
```

This does two things:

1. Runs the default method suite
2. Generates the Scene 282-3 comparison package

#### 8.3 Regenerate only the Scene 282-3 comparison outputs

```powershell
.venv\Scripts\python.exe -m mm5_calib_benchmark.scripts.make_scene_2823_comparison
```

#### 8.4 Reproduce the legacy MAR Scene 282 pipeline

```powershell
.venv\Scripts\python.exe -m mm5_calib_benchmark.scripts.run_legacy_mar_scene282 --mar-mode all --save-level debug
```

Important: this script is not standalone. It expects a sibling repository named `MAR_test` containing `backup_2600.py` and the related assets.

The expected layout is roughly:

```text
.../aMAR/
|- MAR_bianyuan/
`- MAR_test/
```

### 9. Where results are written

The main output root is:

- [mm5_calib_benchmark/outputs/mm5_benchmark](/e:/aa_read_yan/aMAR/MAR_bianyuan/mm5_calib_benchmark/outputs/mm5_benchmark:1)

The most important subfolders are:

1. `method_M*_.../`
   Per-method, per-track outputs. These usually include `calib/`, `metrics/`, `masks/`, `viz/`, and `warped/`.
2. `legacy_mar_scene282_reproduced/`
   Reproduced outputs from the historical MAR workflow
3. `scene_282_3_comparison/`
   Summary figures, ranking tables, metric notes, and MAR history notes
4. `splits/`
   Split files and the index CSV with split labels

### 10. How to interpret the results

If you are reading this project for the first time, a good order is:

1. Read this `README`
2. Read [mar_scholar_compare/README.md](/e:/aa_read_yan/aMAR/MAR_bianyuan/mar_scholar_compare/README.md:1)
3. Explore `mm5_calib_benchmark/outputs/mm5_benchmark/scene_282_3_comparison/`
4. Pay special attention to:
   [scene_282_3_metric_explanation.md](/e:/aa_read_yan/aMAR/MAR_bianyuan/mm5_calib_benchmark/outputs/mm5_benchmark/scene_282_3_comparison/scene_282_3_metric_explanation.md:1)
   [scene_282_3_mar_history_note.md](/e:/aa_read_yan/aMAR/MAR_bianyuan/mm5_calib_benchmark/outputs/mm5_benchmark/scene_282_3_comparison/scene_282_3_mar_history_note.md:1)

The most important metrics are:

1. `pixel_accuracy`
   Overall pixel classification correctness. Often high, but can be overly optimistic because background dominates.
2. `mean_iou`
   More sensitive to whether foreground classes are actually aligned.
3. `boundary_f1`
   Better reflects boundary quality.
4. `keypoint_transfer_error_px`
   More directly reflects geometric misalignment in pixels.

This project should therefore be read using semantic, boundary, and geometric metrics together rather than a single score.

### 11. What is already included in the repository

This repository currently stores not only code, but also a large amount of experiment output. That means you can already inspect:

1. Method implementations
2. Calibration files and YAML assets
3. Generated benchmark outputs
4. Scene 282 comparison figures and Word documents
5. Reproduced legacy MAR outputs

This is convenient for review, but it also makes the repository relatively heavy.

### 12. Current limitations

These are the main constraints new contributors should know immediately:

1. The full MM5 raw dataset is not bundled here
2. Some methods are engineering approximations of paper ideas rather than exact official reproductions
3. `run_legacy_mar_scene282.py` depends on an external sibling `MAR_test` workspace
4. Many path references are currently tied to the author's local machine layout
5. Code, outputs, figures, and reports coexist in one repository, so "source" and "artifact" are easy to confuse at first glance

### 13. Recommended onboarding path for a new contributor

If you are taking over this project, the recommended order is:

1. Read this `README`
2. Understand that `mm5_calib_benchmark/` is the main code, `runs/` is mostly experiment records, and `mar_scholar_compare/` is analysis material
3. Read [mm5_calib_benchmark/pipeline.py](/e:/aa_read_yan/aMAR/MAR_bianyuan/mm5_calib_benchmark/pipeline.py:1) to understand the global workflow
4. Then inspect each method under `methods/`
5. Finally decide whether you want to inspect existing outputs only, or actually rerun the benchmark on a machine with valid MM5 data paths
