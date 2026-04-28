# MM5 Multimodal Registration Workspace

这是一个围绕 MM5 多模态数据集建立的配准、标定和评估工作区。项目的核心目标是：

1. 复现和比较多种 RGB、LWIR、UV、depth 相关的跨模态配准思路。
2. 在不使用 MM5 aligned 图像参与生成的前提下，尽量用用户自己的标定数据和原始传感器数据重建接近原文 aligned 的配准质量。
3. 保留可复现实验记录、指标、可视化面板和阶段性结论，方便后续继续优化或同步到 GitHub。

当前最重要的成果是 `darklight_mm5/calibration_only_method` 下的 Phase25 深度辅助配准方案。它在三张选定暗光样本 `106,104,103` 上达到：

| Method | RGB NCC mean/min | LWIR NCC mean/min | Note |
|---|---:|---:|---|
| Phase24 baseline | `0.9865 / 0.9724` | `0.9182 / 0.9118` | checkerboard-derived LWIR affine |
| Phase25 promoted | `0.9865 / 0.9724` | `0.9321 / 0.9261` | depth-assisted residual registration |
| retained bridge target | - | `0.9233 / 0.9064` | previous MM5 aligned bridge level |

也就是说，Phase25 在这三张样本上已经超过 retained bridge 的 LWIR NCC mean/min，同时保持 RGB aligned 级别不下降。

## 核心原则

Phase25 的生成路径只允许使用：

- `calibration/` 中的标定文件；
- MM5 index 指向的原始 RGB1、LWIR16、depth 图像；
- 原始 calibration-board captures；
- 固定的几何、棋盘格角点、depth 前景边界等从原始数据和标定数据计算出的信息。

Phase25 不允许使用：

- MM5 aligned RGB/T16 作为配准参数来源；
- teacher residual、aligned template、official transform 之类从参考结果反推的参数；
- 用 aligned 图像对每个样本做单独拟合或调参。

MM5 aligned 图像只在最后评估阶段读取，用来报告 NCC、edge distance 等指标。

## 当前最佳方案：Phase25

主入口：

```powershell
python .\darklight_mm5\calibration_only_method\run_phase25_depth_assisted.py --aligned-ids 106,104,103
```

主输出：

```text
darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/
```

推荐阅读：

- [Phase25 子目录 README](darklight_mm5/calibration_only_method/README.md)
- [Phase25 实验报告](darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/phase25_depth_assisted_report.md)
- [Phase25 summary CSV](darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/metrics/phase25_depth_assisted_summary.csv)
- [Phase25 depth registration scores](darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/metrics/phase25_depth_registration_scores.csv)

Phase25 的逻辑是：

1. 保留 Phase21 的 RGB canvas，使 RGB 对 MM5 aligned RGB 的 NCC 保持 `0.9865 / 0.9724`。
2. 使用 Phase23 从 calibration-board corners 得到的 LWIR crop offset `(280,115)`。
3. 使用 Phase24 从 `1848` 个 checkerboard correspondence points 拟合出的 `affine_lmeds` LWIR residual transform。
4. 将 raw depth 裁剪到固定 RGB canvas。
5. 用 `depth < 1000 mm` 提取近景 foreground boundary。
6. 在 `2 px` 半径内搜索一个三张图共享的 LWIR residual translation。
7. 由 depth-boundary-to-LWIR-edge distance 选择 `dx=-2 px, dy=+2 px`。
8. 只用 dense depth projection 填补 residual shift 后产生的新边界无效区。

这一步是真正的 depth-assisted registration：depth 不是只用来补洞，而是参与 residual registration 参数选择。

## 项目目录结构

```text
MAR_bianyuan/
|- calibration/
|- darklight_mm5/
|  |- calibration_only_method/
|  |- run_darklight.py
|  `- legacy/older experiment files and ignored outputs
|- mm5_calib_benchmark/
|- mar_scholar_compare/
|- mm5_ivf/
|- docs/
|- runs/
|- task_plan.md
|- findings.md
|- progress.md
|- .gitignore
`- README.md
```

下面按逻辑介绍每个目录的职责。

## `calibration/`

这里保存项目使用的相机标定文件，是 Phase25 和 benchmark 方法的共同基础。

重要文件：

- `def_stereocalib_THERM.yml`：RGB/LWIR stereo calibration、`R1/R2/P1/P2` 等 rectification 参数。
- `def_thermalcam_ori.yml`：raw thermal camera intrinsics，Phase25 和前置 Phase 中用于 raw LWIR rectification/projection。
- `def_stereocalib_UV.yml`、`def_uvcam_ori.yml`：UV 方向相关标定。
- `calib_device_0.json`、`calib_device_1.json`：设备级标定信息。

这些文件是允许参与生成的标定数据。

## `darklight_mm5/`

这是围绕 MM5 dark/light 样本做配准实验的工作区。当前 GitHub 同步时重点保留 Phase25 相关内容。

### `darklight_mm5/calibration_only_method/`

这是当前最重要的目录，保存“只基于标定数据和原始输入”的 aligned-style 重建方法。

关键文件：

- `run_phase25_depth_assisted.py`：当前最佳 Phase25 主入口。
- `README.md`：Phase25 方案细节、指标和命令。
- `outputs_phase25_depth_assisted/`：当前最佳输出，包含 metrics、panels 和报告。
- `run_phase21_canvas_optimization.py`、`run_phase22_stereo_recalib.py`、`run_phase23_lwir_board_offset.py`、`run_phase24_lwir_board_affine.py`：Phase25 复用的稳定 helper 代码。旧输出已经不再保留，但这些脚本仍被 Phase25 import。
- `diagnose_aligned_canvas.py`、`run_calibration_only.py`：历史诊断和早期 calibration-only baseline，作为研究记录和 helper 来源保留。

旧 Phase 输出目录已经从工作区清理或在 `.gitignore` 中排除；GitHub 同步重点是 Phase25。

### `darklight_mm5/run_darklight.py`

早期 dark/light 注册与评估工具，Phase25 复用了其中的图像读取、归一化、NCC/edge metric、panel 生成等基础函数。

### 旧实验输出和 teacher residual 目录

以下目录属于旧实验或低效果阶段，已被 `.gitignore` 排除，不作为最新 GitHub 同步重点：

- `darklight_mm5/outputs/`
- `darklight_mm5/outputs_calibration_plane*/`
- `darklight_mm5/teacher_residual_method/`
- `darklight_mm5/calibration_only_method/outputs_phase20_canvas/`
- `darklight_mm5/calibration_only_method/outputs_phase21_canvas/`
- `darklight_mm5/calibration_only_method/outputs_phase22_stereo_recalib/`
- `darklight_mm5/calibration_only_method/outputs_phase23_lwir_board_offset/`
- `darklight_mm5/calibration_only_method/outputs_phase24_lwir_board_affine/`

这些旧阶段对分析有价值，但不是当前最佳版本。

## `mm5_calib_benchmark/`

这是更完整的 MM5 多方法 benchmark 框架，用来比较不同跨模态配准/标定方法。

它包含：

- `configs/`：全局配置和各方法配置。
- `methods/`：M0 到 M7 的方法实现，例如 official baseline、Zhang-style calibration、depth bridge、EPnP、MAR edge refine、depth-guided self calibration。
- `eval/`：几何、mask、boundary、NCC 等评估指标。
- `viz/`：可视化工具。
- `scripts/`：生成 split、运行 benchmark、生成 Scene 282 对比图和文档的入口。
- `outputs/mm5_benchmark/`：benchmark 输出和 Scene 282 对比材料。

注意：这个目录当前有很多历史输出和 benchmark 改动。Phase25 同步时不建议直接 `git add .`，避免把无关 benchmark 输出一起提交。

常用入口：

```powershell
python -m mm5_calib_benchmark.scripts.make_splits
python -m mm5_calib_benchmark.scripts.run_all_methods
python -m mm5_calib_benchmark.scripts.make_scene_2823_comparison
```

## `mar_scholar_compare/`

这里保存 Scene 282 相关的论文/历史 MAR 对比材料，更偏分析和展示，不是 Phase25 的主运行路径。

它的作用是帮助理解：

- 历史 MAR 结果与当前 benchmark 结果的差异；
- Scene 282 中不同指标如何解释；
- visual panel、ranking bar、normalized error heatmap 等材料如何阅读。

## `mm5_ivf/`

这是另一个与 MM5 相关的实验/可视化或融合方向工作区。它不是当前 Phase25 配准主线，但作为历史实验记录保留。

## `docs/`

保存协作过程中的设计文档和规格说明，例如之前的 teacher-guided residual alignment 设计文档。

这些文档记录了探索路径，但当前最佳实现仍以 Phase25 为准。

## `runs/`

保存阶段性运行结果、人工分析文件、旧报告或中间产物。它更像实验记录区，而不是主代码入口。

## 根目录规划文件

这些文件是持续协作时的工作记忆：

- `task_plan.md`：阶段计划、当前决策、目标和状态。
- `findings.md`：实验发现、指标结论、失败路径和分析。
- `progress.md`：逐步运行记录、验证命令、清理和同步记录。

如果后续继续优化，建议先读这三个文件再动代码。

## 数据依赖

仓库本身不包含完整 MM5 原始数据。很多 index 路径指向本机数据目录，例如：

```text
D:\a三模数据\MM5_RAW\...
D:\a三模数据\MM5_ALIGNED\...
D:\a三模数据\MM5_CALIBRATION\...
```

要重新运行 Phase25，需要这些路径在当前机器上真实存在。

Phase25 会读取 aligned 图像做评估，但不会用 aligned 图像生成或选择参数。

## 环境依赖

主要 Python 包：

- `numpy`
- `opencv-python`
- `scipy`
- `scikit-image`
- `python-docx`，仅部分文档生成脚本需要

推荐在项目根目录使用已有 `.venv` 或同等 Python 环境运行。

## 推荐使用流程

### 1. 只复现当前最佳 Phase25

```powershell
Set-Location 'E:\aa_read_yan\aMAR\MAR_bianyuan'
python .\darklight_mm5\calibration_only_method\run_phase25_depth_assisted.py --aligned-ids 106,104,103
```

然后查看：

```text
darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/phase25_depth_assisted_report.md
```

### 2. 查看可视化面板

```text
darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/panels/
```

其中 promoted candidate 文件名包含：

```text
phase25_depth_registered_global_shift_depth_fill
```

### 3. 看指标表

```text
darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/metrics/phase25_depth_assisted_summary.csv
```

重点看：

- `eval_rgb_to_mm5_aligned_rgb_ncc_mean`
- `eval_lwir_to_mm5_aligned_t16_ncc_mean`
- `eval_lwir_to_mm5_aligned_t16_ncc_min`
- `eval_lwir_to_mm5_aligned_t16_edge_distance_mean`

## GitHub 同步建议

由于仓库里有大量历史 benchmark 输出和无关实验结果，建议不要直接使用：

```powershell
git add .
```

推荐只同步最新版 Phase25 相关内容：

```powershell
git add .gitignore README.md darklight_mm5/run_darklight.py darklight_mm5/calibration_only_method
git commit -m "Add Phase25 depth-assisted MM5 registration"
git push origin main
```

如果需要同步规划记录，可以额外加入：

```powershell
git add task_plan.md findings.md progress.md
```

## 当前状态总结

- 当前最佳：Phase25 depth-assisted residual registration。
- 当前三张测试图：`106,104,103`。
- RGB 已达到 aligned 级别：`0.9865 / 0.9724`。
- LWIR 已超过 retained bridge：`0.9321 / 0.9261`。
- 旧 Phase 输出已不再作为最新版同步内容。
- 后续若继续优化，应在 Phase25 基础上做小范围验证，而不是回到 teacher/aligned-derived 调参路线。
