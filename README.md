# MM5 多模态配准、Phase29 v9 验收与 FPGA 实时融合项目

本仓库整理了 MM5 RGB/LWIR/depth 多模态配准、可视化验收证据、历史研究版本，以及 DA1501A 单点激光测距辅助 FPGA/HLS 实时融合原型。当前项目已经从早期 Phase24/Phase25 calibration-only 几何基线，迭代到 Phase29 v9 support-gated broad-generalization 验收版本。

当前主线只有一条：

```text
darklight_mm5/calibration_only_method/phase29/
```

Phase29 v9 是当前保留的最佳验收版本。旧版本的关键指标、失败经验和设计理由已经压缩进 README、`task_plan.md`、`findings.md`、`progress.md` 和 `docs/`，不再依赖大体积旧输出目录来记忆项目历史。

## 当前状态

| 项目项 | 当前结论 |
|---|---|
| 当前验收方法 | Phase29 v9 support-gated registration/evidence method |
| 当前代码入口 | `darklight_mm5/calibration_only_method/phase29/run_phase29.py` |
| 当前方法说明 | `darklight_mm5/calibration_only_method/phase29/README.md` |
| 当前保留输出 | v9 `core`、`review`、`broad` 三套输出 |
| 生成和选择边界 | 只使用 calibration、raw RGB、raw LWIR、raw depth、棋盘格几何和 raw/depth support |
| aligned 图像用途 | 只用于 evaluation、heatmap、oracle ceiling 和视觉验收对比 |
| broad 压力集结果 | `18/18` pass，edge mean/max `2.2408 / 2.9714 px` |
| FPGA/HLS 方向 | `peizhun_jiguang/` 中的 Phase25-seeded 单点激光测距辅助原型 |

Phase29 v9 需要诚实描述为 support-gated 注册证据方法。对于反光、背景异常、弱目标和双边缘场景，它会抑制不可靠背景热边缘，输出目标相关的 LWIR 注册证据，而不是声称重建了全场景物理热图。

## 快速入口

| 目标 | 路径 |
|---|---|
| 当前 Phase29 v9 方法说明 | `darklight_mm5/calibration_only_method/phase29/README.md` |
| 当前 Phase29 v9 脚本 | `darklight_mm5/calibration_only_method/phase29/run_phase29.py` |
| broad 验收报告 | `darklight_mm5/calibration_only_method/phase29/outputs_broad_generalization_v9/reports/p29_broad_generalization_report.md` |
| broad 五联图 | `darklight_mm5/calibration_only_method/phase29/outputs_broad_generalization_v9/five_panels/` |
| broad acceptance summary panels | `darklight_mm5/calibration_only_method/phase29/outputs_broad_generalization_v9/acceptance_summary_panels/` |
| broad reliability maps | `darklight_mm5/calibration_only_method/phase29/outputs_broad_generalization_v9/reliability_maps/` |
| strict calibration-only 总说明 | `darklight_mm5/calibration_only_method/README.md` |
| MM5 工作区说明 | `darklight_mm5/README.md` |
| FPGA/HLS 激光辅助原型 | `peizhun_jiguang/README.md` |
| HLS 目录说明 | `peizhun_jiguang/hls/README.md` |
| 文档索引 | `docs/README.md` |
| 目录体检记录 | `docs/project_structure_review_2026-05-08.md` |
| 阶段计划 | `task_plan.md` |
| 技术发现 | `findings.md` |
| 执行记录 | `progress.md` |

## 当前验收指标

Phase29 v9 使用 `support-v9` candidate grid 和 selector `v6`。每个样本生成 `22` 个 strict candidates，最终选择只基于 raw/depth 证据，不读取 aligned RGB/T16 作为选择依据。

| Profile | 样本范围 | Result |
|---|---|---:|
| core | `106,104,103` | `3/3` pass, edge mean/max `1.7249 / 1.8926 px`, improved/regressed `2 / 0` |
| review | `2,23,103,106,273,291,302` | `7/7` pass, edge mean/max `1.6379 / 2.7155 px`, improved/regressed `3 / 0` |
| broad | 18-sample pressure set | `18/18` pass, edge mean/max `2.2408 / 2.9714 px`, improved/regressed `8 / 0` |

之前 broad pressure set 中最难的样本已经进入通过范围：

| 样本 | v9 selected mode | edge distance |
|---|---|---:|
| `050_seq332` | `p29_v9_depth_thermal` | `1.5848 px` |
| `110_seq396` | `p29_v9_target_only` | `2.2281 px` |
| `123_seq409` | `p29_v9_target_silhouette` | `1.2253 px` |
| `187_seq473` | `p29_v9_depth_thermal` | `2.7905 px` |

## 方法边界

当前 Phase29 v9 生成和选择阶段允许使用：

- `calibration/` 中的用户标定文件；
- MM5 index 指向的 raw RGB1、raw LWIR16、raw depth 图像；
- 原始 calibration-board captures；
- calibration-derived board correspondences；
- depth foreground/support masks；
- anti-ghost masks、risk maps 和 raw/depth-only selector gates；
- 固定几何模型、固定输出 canvas、固定 crop/affine 参数。

当前 Phase29 v9 生成和选择阶段不允许使用：

- MM5 aligned RGB/T16 作为参数来源；
- official aligned transform、aligned template、teacher residual；
- 读取 aligned 图像后针对单张样本做 per-sample fitting 或调参；
- 用 aligned 指标选择最终生成候选。

aligned 图像只在评估阶段读取，用于报告 NCC、edge distance、heatmap、oracle ceiling 和视觉验收对比。

## 仓库结构

当前工作目录名是 `MAR_bianyuan`，Git 远端仓库名仍是 `MM5_test`。README 中使用 `MM5_test/` 作为项目逻辑名，实际本地路径可以是任意克隆目录。

```text
MM5_test/
|- calibration/
|- darklight_mm5/
|  |- calibration_only_method/
|  |  `- phase29/
|  |- docs/
|  `- teacher_residual_method/
|- docs/
|- mar_scholar_compare/
|- mm5_calib_benchmark/
|- mm5_ivf/
|- peizhun_jiguang/
|- runs/
|- task_plan.md
|- findings.md
|- progress.md
`- README.md
```

| 文件夹 | 当前职责 |
|---|---|
| `calibration/` | 用户相机标定文件，是 MM5 配准、benchmark 和 FPGA 参数导出的共同基础。 |
| `darklight_mm5/` | MM5 RGB/LWIR/depth 配准、融合和评估主工作区。当前主线在 `calibration_only_method/phase29/`。 |
| `docs/` | 项目文档索引、设计记录、目录体检、整理清单和一次性工具。 |
| `mar_scholar_compare/` | Scene 282 和 MAR 论文式对比材料，作为历史分析和展示归档保留。 |
| `mm5_calib_benchmark/` | MM5 多方法 benchmark 框架和 split index。清理后 active output 只保留 `splits/index_with_splits.csv`。 |
| `mm5_ivf/` | MM5 历史实验代码，包含数据构建、训练、融合和可视化探索，不是当前验收主线。 |
| `peizhun_jiguang/` | DA1501A 单点激光测距辅助 FPGA/HLS IP 原型，是实时化方向。 |
| `runs/` | 历史运行结果、人工分析图、Word 报告和阶段性输出。 |

本地环境目录如 `.venv/`、`.git_ssh/`、`.superpowers/`、HLS build/cache、Python `__pycache__` 不作为项目内容同步。

## 目录命名策略

这次整理没有直接重命名真实目录，原因是 Phase29 输出目录、Markdown 报告、复现命令和脚本参数已经互相引用。直接改名会破坏可复现性。当前命名策略如下：

| 命名 | 保留原因 |
|---|---|
| `darklight_mm5` | 保留早期暗光 MM5 主工作区名称，内部 README 已说明当前不只包含暗光样本。 |
| `calibration_only_method` | 明确表达当前 strict boundary：生成和选择不能使用 aligned 图像。 |
| `phase29` | 当前验收主线，Phase28 helper 已整合到该目录，保持历史阶段号可追踪。 |
| `outputs_core_generalization_v9` | 核心三样本验收输出，名称与复现命令一致。 |
| `outputs_review_generalization_v9` | 七样本代表性 review 输出，名称与报告一致。 |
| `outputs_broad_generalization_v9` | 十八样本压力集验收输出，当前最重要的视觉证据目录。 |
| `peizhun_jiguang` | 现有 FPGA/HLS 激光辅助方向目录，保留中文拼音命名以避免脚本路径失效。 |
| `mm5_ivf` | 历史探索工作区，保留原实验线名称。 |
| `mar_scholar_compare` | 历史论文式对比材料，保留描述性名称。 |

如果后续要做真实目录重命名，建议单独开一个阶段，先生成重命名 manifest，再统一更新 Python imports、README、报告引用、PowerShell 脚本和 Git 历史说明。

## 复现当前 Phase29 v9

从仓库根目录运行：

```powershell
python .\darklight_mm5\calibration_only_method\phase29\run_phase29.py --run-profile core --candidate-grid support-v9 --version-label v9 --selector-version v6 --output .\darklight_mm5\calibration_only_method\phase29\outputs_core_generalization_v9 --report-level research --save-selector-debug
python .\darklight_mm5\calibration_only_method\phase29\run_phase29.py --run-profile review --candidate-grid support-v9 --version-label v9 --selector-version v6 --output .\darklight_mm5\calibration_only_method\phase29\outputs_review_generalization_v9 --report-level research --save-selector-debug
python .\darklight_mm5\calibration_only_method\phase29\run_phase29.py --run-profile broad --candidate-grid support-v9 --version-label v9 --selector-version v6 --output .\darklight_mm5\calibration_only_method\phase29\outputs_broad_generalization_v9 --report-level research --save-selector-debug
```

验证：

```powershell
python -m py_compile .\darklight_mm5\calibration_only_method\phase29\run_phase29.py
python -m json.tool .\darklight_mm5\calibration_only_method\phase29\outputs_core_generalization_v9\metrics\p29_best.json
python -m json.tool .\darklight_mm5\calibration_only_method\phase29\outputs_review_generalization_v9\metrics\p29_best.json
python -m json.tool .\darklight_mm5\calibration_only_method\phase29\outputs_broad_generalization_v9\metrics\p29_best.json
```

## Phase29 v9 输出结构

每套 v9 输出目录包含同一组证据子目录，便于 core、review、broad 横向对比。

| 子目录 | 用途 |
|---|---|
| `five_panels/` | 主五联图：generated RGB、selected LWIR、轮廓、融合、risk map。 |
| `acceptance_summary_panels/` | 每个样本的一页式验收摘要图。 |
| `selected_clean_lwir/` | 最终选中的 clean LWIR 或 support-gated LWIR evidence。 |
| `selected_registered_lwir/` | 注册后的 LWIR 输出。 |
| `selected_fusion_review/` | anti-ghost fusion review。 |
| `contour_overlays/` | RGB/LWIR/depth 轮廓对齐叠加。 |
| `edge_error_heatmaps/` | 评估用边缘误差热图。 |
| `hard_ceiling_panels/` | selected-vs-best-strict evidence。 |
| `oracle_ceiling_panels/` | evaluation-only oracle 对照，不能作为生成依据。 |
| `reliability_maps/` | reliability/risk overlay。 |
| `selector_debug/` | 每个候选的 raw/depth 选择证据。 |
| `metrics/` | CSV/JSON 指标，包括 `p29_best.json`。 |
| `reports/` | Markdown 验收或研究报告。 |

## 版本记忆

旧版本输出体已经清理，但核心结论仍保留在文档中。

| 阶段 | 结果与意义 |
|---|---|
| Phase24 board-affine | 棋盘格推导的 LWIR affine baseline；LWIR NCC mean/min `0.9182 / 0.9118`。 |
| Phase25 promoted | 三张暗光样本 `106,104,103` 上，LWIR NCC mean/min `0.9321 / 0.9261`，edge mean `14.8499 px`；它是后续几何和 FPGA/HLS 的基础。 |
| Phase28 | 稳定视觉 baseline；core/review 通过，broad `11/18`，edge mean/max `2.8978 / 5.4010 px`。 |
| Phase29 v3 | 第一版 honest broad-generalization；broad `13/18`。 |
| Phase29 v4/v5 | broad 提升到 `15/18`；v5 加入 reliability labels、hard-ceiling evidence 和 selector explanation。 |
| Phase29 v6 acceptance-lite | v9 前最佳 compact selector；broad `15/18`，edge mean/max `2.6283 / 4.7164 px`，`13` candidates/sample。 |
| Phase29 v7/v8 | research probes；大候选池或 component/depth projection 没有稳定解决困难样本，未被提升为验收版本。 |
| Phase29 v9 | 当前验收版本；core `3/3`、review `7/7`、broad `18/18`，broad edge mean/max `2.2408 / 2.9714 px`。 |

旧 calibration-plane、teacher residual、Phase28/Phase25 输出体、benchmark method outputs、HLS 本地 build 产物已经从 active workspace 清理。它们的核心结论保留在 README、`task_plan.md`、`findings.md` 和 `progress.md`。

## FPGA / HLS IP 方向

FPGA 方向不是直接复制 dense depth 图像流程，而是把离线几何结果转换成：

```text
single laser range -> range-bin LUT -> fixed-point LWIR warp -> RGB/LWIR fusion
```

目标 HLS IP top：

```text
phase25_laser_register_fuse_ip_top
```

这个单 IP 集成了：

- DA1501A Protocol 1 接收帧解析；
- checksum、reserved byte、valid/stale/blind-zone/fallback 状态检查；
- distance、range age 和 status debug 输出；
- Phase24 fallback geometry；
- Phase25-seeded range-bin 参数选择；
- fixed-point LWIR inverse affine warp；
- packed RGB/LWIR fusion。

历史 HLS synthesis 参考结果：

| Top | Part | Estimated clock | Fmax | Latency | Resources |
|---|---|---:|---:|---:|---|
| `phase25_laser_register_fuse_ip_top` | `xczu15eg-ffvb1156-2-e` | `7.300 ns` | `136.99 MHz` | `307244-307310 cycles`, about `3.072-3.073 ms` per `640x480` frame | `0 BRAM18K`, `8 DSP`, `4281 FF`, `7302 LUT`, `0 URAM` |

注意：当前 LUT 仍是 Phase25 seed/fallback 原型。真实硬件精度需要完成 DA1501A 与 RGB/LWIR 相机的机械安装、laser-to-camera 标定和 range-bin 实测。本地 HLS build 目录已经清理，源码、脚本和文档仍保留。

## 复现 FPGA/HLS 辅助流程

导出 range-bin LUT：

```powershell
python .\peizhun_jiguang\scripts\export_laser_lut.py `
  --config .\peizhun_jiguang\config\laser_registration_params.json `
  --output-dir .\peizhun_jiguang\generated
```

验证 HLS C++ 行为：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\peizhun_jiguang\hls\run_manual_clang_check.ps1
```

运行 HLS synthesis：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\peizhun_jiguang\hls\run_vitis_hls_synth.ps1
```

## 数据与环境说明

仓库保存代码、标定、关键输出和分析材料。完整 MM5 原始数据通常仍位于本机数据盘。重新运行 Phase29 时，需要 index 指向的 raw RGB、raw LWIR、raw depth 和 evaluation-only aligned 图像实际存在。

主要 Python 依赖：

- `numpy`
- `opencv-python`
- `scipy`
- `scikit-image`
- `python-docx`

FPGA/HLS 相关：

- 本机历史环境发现 Vitis HLS 2022.1。
- 临时目标器件族为 `xczu15eg`。
- 历史综合 full part 为 `xczu15eg-ffvb1156-2-e`。
- HLS 本地生成工程和 build 产物不纳入 GitHub 同步。

## 维护和清理边界

应该保留：

- `calibration/` 标定文件；
- `darklight_mm5/calibration_only_method/phase29/` 源码和 v9 core/review/broad 输出；
- `mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv`；
- `peizhun_jiguang/` 源码、配置、LUT 和 HLS 文档；
- `runs/` 中的历史报告和人工分析材料；
- `task_plan.md`、`findings.md`、`progress.md`；
- `docs/` 中的设计记录、体检记录和整理工具。

可以视为临时或本地内容：

- `.venv/`；
- `.git_ssh/`；
- `.superpowers/`；
- Python `__pycache__/`；
- HLS build/project 输出；
- 重新生成的非 v9 临时输出目录。

## 编码提示

README 使用 UTF-8。PowerShell 读取中文 Markdown 时建议显式指定编码：

```powershell
Get-Content -Raw -Encoding UTF8 .\README.md
```

如果不指定编码，部分 Windows shell 会把中文显示成乱码，但文件本身仍可能是正常 UTF-8。

## 当前一句话总结

这个仓库现在保留了一条清晰主线：以 Phase29 v9 作为 MM5 raw/calibration/depth-only 的当前验收配准方案，同时保留 DA1501A 单点激光测距辅助 FPGA/HLS 原型作为实时化方向。旧版本的指标和经验已经整合进文档，生成物则从 active workspace 中清理，便于后续验收、复现和 GitHub 同步。
