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

主流程代码在 [mm5_calib_benchmark/pipeline.py](mm5_calib_benchmark/pipeline.py)，配置加载逻辑在 [mm5_calib_benchmark/config.py](mm5_calib_benchmark/config.py)。

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

从现有 `scene_282_3_metric_explanation.md` 的结果摘要看，按 `normalized_overall_region_error` 排名时 `M7` 当前最低，但它伴随非常低的 `valid_warp_coverage`；如果按“误差、覆盖率和语义质量一起看”的更稳健标准，`M5`、`M4`、`M6` 仍然是当前更均衡的第一梯队。

### 6. 输入数据是怎样组织的

这个项目不是“纯代码仓库”，它依赖 MM5 的原始数据、对齐数据、标定目录和标注文件。

当前索引文件位于：

- [mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv](mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv)

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

- [mm5_calib_benchmark/outputs/mm5_benchmark](mm5_calib_benchmark/outputs/mm5_benchmark)

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
2. 再看 [mar_scholar_compare/README.md](mar_scholar_compare/README.md)
3. 再看 `mm5_calib_benchmark/outputs/mm5_benchmark/scene_282_3_comparison/`
4. 重点看下面两个说明文件：
   [scene_282_3_metric_explanation.md](mm5_calib_benchmark/outputs/mm5_benchmark/scene_282_3_comparison/scene_282_3_metric_explanation.md)
   [scene_282_3_mar_history_note.md](mm5_calib_benchmark/outputs/mm5_benchmark/scene_282_3_comparison/scene_282_3_mar_history_note.md)

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
3. 先阅读 [mm5_calib_benchmark/pipeline.py](mm5_calib_benchmark/pipeline.py)，了解总流程
4. 再看 `methods/` 下每个 `run.py`
5. 最后根据自己机器上的数据路径，决定是“只复查现有结果”，还是“重新跑 benchmark”

### 14. Scene 282 核心对比图怎么读

说明：

1. 当前图内标题统一使用英文，不是因为仓库只支持英文，而是因为 OpenCV 默认绘图字体对中文支持不稳定，直接画中文会出现 `???`。
2. 详细中文解释以 `README` 和 `mm5_calib_benchmark/outputs/mm5_benchmark/scene_282_3_comparison/` 下的说明文件为准。
3. 当前主热成像对比图聚焦 `M1 / M2 / M4 / M5 / M6 / M7` 六个 thermal 方法。
4. `M0` 仍然保留在 benchmark 的 per-method 输出目录中，`M3` 是 UV-only 方法，因此不出现在主 thermal 排序图里。

| 产物 | 显示效果 | 建议重点看 | 重要对比参数 |
| --- | --- | --- | --- |
| `scene_282_3_all_methods_comparison.png` | 第一行给出 RGB 源图、thermal 目标图、thermal GT；下面每个方法一行，分别有 prediction、contours、error 三张子图 | `prediction` 看整体区域是否压到正确目标上，`contours` 看边界是否贴边，`error` 看错误来自漏检、误检还是类别混淆 | `mean_iou`、`boundary_f1`、`pixel_accuracy`、`valid_warp_coverage`、`overall_region_error_px` |
| `scene_282_3_thermal_method_ranking_bar.png` | 用 test 集汇总结果给六个 thermal 方法排总榜，并把多个指标放在同一顺序下对比 | 先看 `normalized_overall_region_error` 和 `overall_region_error_px` 判断整体校准精度，再看 `mean_iou` / `pixel_accuracy` 判断语义质量 | `normalized_overall_region_error`、`overall_region_error_px`、`mean_iou`、`pixel_accuracy`、`keypoint_transfer_error_px` |
| `scene_282_3_thermal_per_scene_normalized_error.png` | 每一行对应一张 test 图片、每一列对应一个 thermal 方法，单元格里直接列出该图的归一化整体区域误差 | 看某个方法是不是只靠少数样本拉高平均分，也看哪些图片是真正难例 | `normalized_overall_region_error`、`best_method`、`scene wins` |
| `scene_282_3_mar_history_panel.png` | 并排展示历史 MAR 工程版 / 论文版的 raw 与 aligned 结果，并附带几何诊断信息 | 看历史 MAR 在不同 GT 定义下的表现差异，以及几何假设是否成立 | `pixel_accuracy`、`mean_iou`、`binary_iou_foreground`、`geometry_baseline_ok`、`real_projected_ratio_on_redistort` |
| `scene_282_3_metric_explanation.md` | 指标专门说明文件 | 当你看到 `pixel_accuracy` 很高但图像效果一般时，先回来看这里 | `pixel_accuracy`、`mean_iou`、`boundary_f1`、`overall_region_error_px`、`normalized_overall_region_error` |
| `scene_282_3_figure_guide.md` | 图像阅读说明索引 | 当你不确定“某张图是看效果还是看参数”时，先看这份导读 | 各图对应的重点指标与使用场景 |

主对比图中的误差图颜色说明如下：

1. 绿色：预测正确
2. 蓝色：误检前景（GT 是背景）
3. 橙色：漏掉 GT 前景
4. 紫色：前景类别错分
5. 黑色：无效投影区域

推荐阅读顺序：

1. 先看 `scene_282_3_all_methods_comparison.png`
2. 再看 `scene_282_3_thermal_method_ranking_bar.png`
3. 然后看 `scene_282_3_thermal_per_scene_normalized_error.png`
4. 最后看 `scene_282_3_mar_history_panel.png` 与各类说明 `.md`

### 15. 参数总览：这些参数分别是什么意思

#### 15.1 评价指标参数

| 参数 | 含义 | 趋势 | 更适合回答什么问题 |
| --- | --- | --- | --- |
| `pixel_accuracy` | 所有有效评测像素里，分类完全正确的比例 | 越大越好 | “整体分类看起来对不对” |
| `mean_iou` | 各类别 `IoU` 的平均值，更关注前景和类别区域是否真的重合 | 越大越好 | “区域有没有真正对齐” |
| `mean_pixel_accuracy` | 对每个类别单独算像素正确率后再平均 | 越大越好 | “小类是否被大背景掩盖” |
| `freq_iou` | 频率加权的 IoU | 越大越好 | “按类别出现频率加权后的整体效果” |
| `boundary_f1` | 预测边界与 GT 边界的一致性 | 越大越好 | “边界有没有贴边、是否有一圈错位” |
| `keypoint_transfer_error_px` | 边界/关键点层面的像素转移误差 | 越小越好 | “几何误差大概是多少像素” |
| `overall_region_error_px` | 整个前景区域的双向平均距离误差 | 越小越好 | “整体区域偏了多少像素” |
| `normalized_overall_region_error` | `overall_region_error_px / image_diagonal_px`，把不同分辨率结果归一化 | 越小越好 | “跨模态、跨分辨率的主对比指标” |
| `valid_warp_coverage` | 有效投影区域占目标图像的比例 | 越大越好 | “投影是否覆盖得足够完整” |
| `checkerboard_corner_rmse_px` | 棋盘格角点重投影均方根误差 | 越小越好 | “标定本身是否稳定” |
| `mutual_information` | 源图与目标图在有效区域的统计相关性 | 越大越好 | “跨模态图像内容是否更加一致” |
| `ntg` | Normalized Total Gradient，梯度结构一致性分数 | 越大越好 | “边缘/纹理结构是否更对齐” |
| `num_test_scenes` | test 集纳入统计的场景数量 | 越大越稳 | “这个汇总结果覆盖了多少场景” |

对当前仓库而言，推荐的主指标优先级是：

1. `normalized_overall_region_error`
2. `overall_region_error_px`
3. `mean_iou`
4. `boundary_f1`
5. `pixel_accuracy`

#### 15.2 通用运行参数

这些参数来自 `mm5_calib_benchmark/configs/default.yaml` 或配置加载逻辑：

| 参数 | 位置 | 含义 |
| --- | --- | --- |
| `runtime.seed` | `configs/default.yaml` | 随机采样、训练场景抽样等流程的固定随机种子 |
| `runtime.plane_depth_mm` | `configs/default.yaml` | 平面单应近似时假设的参考平面深度 |
| `runtime.max_test_scenes` | `configs/default.yaml` | 跑 benchmark 时最多纳入的 test 场景数 |
| `outputs.root` | `configs/default.yaml` | benchmark 统一输出根目录 |
| `outputs.index_csv` | `configs/default.yaml` | MM5 索引 CSV 路径 |
| `outputs.splits_dir` | `configs/default.yaml` | `train / val / test` 划分输出目录 |
| `paths.workspace_calibration` | `config.py` 动态注入 | 工作区下 `calibration/` 的绝对路径 |

#### 15.3 通用 `scene_tune` 参数

这些参数由 [mm5_calib_benchmark/methods/alignment.py](mm5_calib_benchmark/methods/alignment.py) 统一解析，多个方法共享：

| 参数 | 含义 |
| --- | --- |
| `coarse_radius_px` | 粗搜索阶段允许的平移搜索半径 |
| `coarse_step_px` | 粗搜索阶段的平移步长 |
| `fine_radius_px` | 细搜索阶段围绕粗搜索最优点继续搜索的半径 |
| `fine_step_px` | 细搜索阶段的平移步长 |
| `coarse_scales` | 粗搜索阶段尝试的缩放系数列表 |
| `coarse_angles_deg` | 粗搜索阶段尝试的旋转角度列表 |
| `fine_scale_delta` | 细搜索阶段围绕当前最优尺度做的小范围扰动 |
| `fine_angle_delta_deg` | 细搜索阶段围绕当前最优角度做的小范围扰动 |
| `edge_weight` | 轮廓贴边分数在综合打分中的权重 |
| `mi_weight` | 互信息分数在综合打分中的权重 |
| `coverage_weight` | 覆盖率变化惩罚的权重 |

说明：

1. 并不是所有方法的 YAML 都显式写出了全部 `scene_tune` 参数。
2. 如果方法 YAML 没写，代码会回落到 `methods/alignment.py` 里的默认值。
3. 所以读参数时，要同时看方法 YAML 和 `scene_tune_kwargs(...)` 里的默认配置。

### 16. 每个方法是什么，以及关键参数是什么意思

#### M0 MM5 Official

1. 核心思路：直接使用 MM5 官方 stereo / 标定结果，再做平面单应投影和场景级微调。
2. 适用定位：作为“官方标定直接拿来用”的基线。
3. 关键参数：主要是 `scene_tune.*`。它决定在单应投影之后，还允许做多大范围的平移、缩放、旋转细调。
4. 方法特点：实现简单、基线清楚，但如果官方标定和当前场景存在偏差，通常需要后续 `scene_tune` 才能追上更强方法。

#### M1 Zhang

1. 核心思路：用棋盘格观测做 OpenCV / Zhang 风格 stereo 标定，然后再做平面单应投影和场景微调。
2. 关键参数：
   `square_size_mm`：棋盘格单格的物理边长。
   `scene_tune.*`：控制场景级搜索范围。
3. 方法特点：比 M0 更偏“自己重建标定”，适合作为“经典棋盘格标定 + 平面对齐”的标准基线。
4. 风险点：如果棋盘格标定本身不稳，后续 scene tuning 只能补一部分误差。

#### M2 Su2025 XoFTR Fallback

1. 核心思路：先使用保存下来的 M1/官方标定得到基础单应结果，再做跨模态特征匹配和仿射修正，最后再做场景微调。
2. 关键参数：
   `feature_refine.use_clahe`：是否先做 CLAHE 增强，帮助跨模态特征稳定。
   `feature_refine.sift_nfeatures` / `orb_nfeatures`：特征点上限。
   `feature_refine.ratio_test`：特征匹配的 Lowe ratio 阈值。
   `feature_refine.max_matches` / `min_matches`：参与仿射估计的匹配数量上下限。
   `feature_refine.ransac_reproj_threshold` / `max_iters`：RANSAC 仿射估计的鲁棒性参数。
   `feature_refine.scale_min` / `scale_max` / `max_translation_px`：仿射修正的保护阈值，防止估计出离谱结果。
   `scene_tune.*`：在特征仿射之后再做一轮稳定化微调。
3. 方法特点：适合“纹理/边缘仍然能提供一定跨模态匹配信息”的场景。
4. 风险点：特征点不足或跨模态差异太大时，可能直接回退，不一定总能带来提升。

#### M3 Jay2025 SGM

1. 核心思路：先做基础单应，再用稠密光流把源图向目标图做更局部的非刚性注册。
2. 当前实现定位：仓库里是一个 registration fallback，主要服务于 UV 方向。
3. 关键参数：
   `flow.pyr_scale` / `levels` / `winsize` / `iterations` / `poly_n` / `poly_sigma`：Farneback 光流的标准控制项。
   `flow.max_flow_px`：对光流位移做上限裁剪，防止局部失控。
   `scene_tune.*`：在光流修正后再做轻量全局微调。
4. 方法特点：能处理部分局部形变，但对跨模态纹理一致性比较敏感。
5. 限制：当前主图不展示它，是因为它主要是 UV-only 方法。

#### M4 Muhovic DepthBridge

1. 核心思路：把深度投影引入到跨模态对齐里，用深度把 RGB 源图上的目标投到目标模态平面，再做 scene tuning。
2. 关键参数：
   `scene_tune.*`：最终全局微调范围。
   当前实现里另外固定使用了 `fill_holes=True`、`fill_distance_px=10.0`、`support_dilate_ksize=11`、`splat_radius=1` 这些内部默认值，它们控制深度投影后的空洞填补和支持区域扩张。
3. 方法特点：比纯平面单应更适合处理存在深度起伏的前景目标。
4. 风险点：如果深度图本身稀疏或噪声大，投影覆盖率和边界稳定性都会受影响。

#### M5 EPnP Baseline

1. 核心思路：先用棋盘格标定结果建立基础几何，再通过 EPnP 估计更强的相对位姿，然后把深度投影作为场景对齐主干。
2. 关键参数：
   `square_size_mm`：棋盘格物理尺寸。
   `scene_tune.*`：投影后的全局微调范围。
   当前实现里固定使用 `fill_distance_px=6.0`、`support_dilate_ksize=7`、`splat_radius=0`，比 M4 更强调几何位姿本身的质量，而不是强行扩张投影。
3. 方法特点：通常是一个较强、较稳的工程基线。
4. 风险点：如果 EPnP 位姿估计受棋盘格噪声影响，整体性能会直接波动。

#### M6 MAR Edge Refine

1. 核心思路：以 M4 的深度桥接结果为基础，再叠加 MAR 风格的边界细化。
2. 当前实现：主要由 [mm5_calib_benchmark/mar_edge_refine.py](mm5_calib_benchmark/mar_edge_refine.py) 负责，内部会做 scene tuning、边界带提取、random walker 细化以及保护性回退。
3. 关键参数：
   `scene_tune.*`：基础几何结果的全局微调。
   当前实现没有单独暴露很多 YAML 级参数，但代码里有关键保护阈值，例如前景面积比、`binary_iou` 下限、连通域数量变化上限等，用来防止“边界细化越修越坏”。
4. 方法特点：边界通常比单纯的深度投影更干净，更适合强调轮廓质量的场景。
5. 风险点：如果 target 边缘本身很弱，边界细化可能保守甚至回退。

#### M7 Depth Guided Self Calibration

1. 核心思路：先做全局姿态自校准，再做深度引导投影、场景微调和边界吸附，是当前仓库里最完整的一条 thermal 路线。
2. 这是 thermal-only 方法。
3. 关键参数：
   `train_refine_scene_count`：用于全局姿态自校准打分的训练场景数量。
   `pose_delta_deg` / `pose_delta_mm`：全局姿态搜索时允许的旋转和平移扰动范围。
   `splat_radius`：深度点投影时的扩散半径。
   `hole_fill_max_dist_px`：投影后空洞允许最近邻补洞的最大距离。
   `support_dilate_ksize`：支持区域膨胀核大小，决定允许补洞的范围。
   `band_width_px`：边界吸附时边界带宽度。
   `quality_gate_min_projected_ratio`：投影覆盖率过低时的质量门槛。
   `quality_gate_min_score_gain`：全局姿态优化后，只有当自监督得分提升超过这个阈值才接受 refined pose。
   `scene_tune.*`：在深度投影之后做的全局轻量搜索。
4. 方法特点：当前最像“完整方案”，在整体区域误差和语义质量上通常都比较强。
5. 风险点：链路长、参数也最多，所以调参和解释成本都高于其他方法。

### 17. 整个流程怎么操作

如果你要从“准备数据”一路走到“看懂结果”，建议按下面流程操作：

#### 17.1 准备数据与标定资源

1. 确保 `index_with_splits.csv` 或 `mm5_index.csv` 能正确指向本机的 MM5 原始数据路径。
2. 确保 `calibration/` 下存在所需的 stereo 标定文件。
3. 如果要复现历史 MAR，还要保证当前仓库同级目录存在 `MAR_test/backup_2600.py`。

#### 17.2 生成数据划分

```powershell
.venv\Scripts\python.exe -m mm5_calib_benchmark.scripts.make_splits
```

这一步会生成 benchmark 使用的 `train / val / test` 划分结果。

#### 17.3 运行完整 benchmark

```powershell
.venv\Scripts\python.exe -m mm5_calib_benchmark.scripts.run_all_methods
```

这一步会做两件事：

1. 运行 `DEFAULT_SUITE` 里的多方法 benchmark
2. 顺便生成 Scene 282 的核心对比产物

`DEFAULT_SUITE` 当前包含：

1. `M0 thermal`
2. `M0 uv`
3. `M1 thermal`
4. `M1 uv`
5. `M5 thermal`
6. `M5 uv`
7. `M2 thermal`
8. `M4 thermal`
9. `M4 uv`
10. `M3 uv`
11. `M6 thermal`
12. `M7 thermal`

#### 17.4 只重生成 Scene 282 对比材料

如果 benchmark 主结果已经在本地存在，而你只想刷新汇总图和说明文件，可以执行：

```powershell
.venv\Scripts\python.exe -m mm5_calib_benchmark.scripts.make_scene_2823_comparison
```

这一步会刷新：

1. `scene_282_3_all_methods_comparison.png`
2. `scene_282_3_thermal_method_ranking_bar.png`
3. `scene_282_3_metrics.csv`
4. `scene_282_3_metric_explanation.md`
5. `scene_282_3_figure_guide.md`
6. `scene_282_3_thermal_per_scene_normalized_error.csv`
7. `scene_282_3_thermal_per_scene_normalized_error.png`
8. `scene_282_3_thermal_per_scene_normalized_error_note.md`
9. 以及历史 MAR 相关汇总图和说明

#### 17.5 查看 per-method 输出

每个方法各自的输出目录通常包含：

1. `calib/`：标定或最终位姿结果
2. `metrics/`：per-scene、per-class 和 summary 指标
3. `masks/`：预测 mask、GT mask、误差图
4. `viz/`：轮廓图、热力图、阶段图
5. `warped/`：投影后的图像与 overlay

如果你要解释“为什么某个方法分数高 / 低”，不要只看 `summary.json`，而要把 `viz/`、`masks/` 和 `warped/` 一起看。

#### 17.6 阅读结果的最短路径

最短建议路径如下：

1. 先看这份 `README`
2. 再看 `scene_282_3_figure_guide.md`
3. 然后看 `scene_282_3_all_methods_comparison.png`
4. 再看 `scene_282_3_thermal_method_ranking_bar.png`
5. 最后用 `scene_282_3_metric_explanation.md` 和 `scene_282_3_thermal_per_scene_normalized_error_note.md` 回头解释各指标

#### 17.7 什么时候该看哪一类指标

1. 如果你想知道“整体区域有没有对齐”，优先看 `normalized_overall_region_error` 和 `overall_region_error_px`
2. 如果你想知道“语义类别有没有对准”，优先看 `mean_iou`
3. 如果你想知道“边界是不是贴边”，优先看 `boundary_f1`
4. 如果你想知道“投影是不是只对上了一小块”，优先看 `valid_warp_coverage`
5. 如果你想知道“棋盘格标定本身稳不稳”，优先看 `checkerboard_corner_rmse_px`

---

## English Version

The Chinese section above contains the most detailed figure guide, parameter glossary, method-by-method explanation, and end-to-end workflow notes for this repository.

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

The central orchestration logic lives in [mm5_calib_benchmark/pipeline.py](mm5_calib_benchmark/pipeline.py), and configuration loading lives in [mm5_calib_benchmark/config.py](mm5_calib_benchmark/config.py).

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

Based on the current benchmark summary, `M7` currently has the lowest `normalized_overall_region_error`, but it does so with very low `valid_warp_coverage`; if you judge by a more balanced combination of error, coverage, and semantic quality, `M5`, `M4`, and `M6` remain the stronger top tier on the current test subset.

### 6. Input data expectations

This is not a self-contained toy repository. It depends on MM5 raw data, aligned reference data, annotations, calibration directories, and an index CSV describing all paths.

The active index file is:

- [mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv](mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv)

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

- [mm5_calib_benchmark/outputs/mm5_benchmark](mm5_calib_benchmark/outputs/mm5_benchmark)

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
2. Read [mar_scholar_compare/README.md](mar_scholar_compare/README.md)
3. Explore `mm5_calib_benchmark/outputs/mm5_benchmark/scene_282_3_comparison/`
4. Pay special attention to:
   [scene_282_3_metric_explanation.md](mm5_calib_benchmark/outputs/mm5_benchmark/scene_282_3_comparison/scene_282_3_metric_explanation.md)
   [scene_282_3_mar_history_note.md](mm5_calib_benchmark/outputs/mm5_benchmark/scene_282_3_comparison/scene_282_3_mar_history_note.md)

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
3. Read [mm5_calib_benchmark/pipeline.py](mm5_calib_benchmark/pipeline.py) to understand the global workflow
4. Then inspect each method under `methods/`
5. Finally decide whether you want to inspect existing outputs only, or actually rerun the benchmark on a machine with valid MM5 data paths
