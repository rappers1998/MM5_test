# MM5 多模态配准与 FPGA 实时融合项目

本仓库整理了 MM5 RGB/LWIR/depth 多模态配准、可视化验收证据，以及 DA1501A 单点激光测距辅助 FPGA/HLS 实时融合原型。项目经历了从早期 Phase25 calibration-only depth-assisted registration，到当前 Phase29 v9 support-gated broad-generalization acceptance 的完整迭代。

当前验收主线是：

```text
darklight_mm5/calibration_only_method/phase29/
```

## 当前验收版本

Phase29 v9 是当前保留的最佳验收版本。它使用 `support-v9` 候选网格和 selector `v6`，生成和选择阶段只使用：

- 用户标定文件；
- raw RGB1、raw LWIR16、raw depth；
- 原始棋盘格采集和 calibration-board geometry；
- raw/depth support masks、anti-ghost masks、risk maps。

MM5 aligned RGB/T16 只用于 evaluation、heatmap、oracle ceiling 和验收面板，不用于生成参数、candidate selection、teacher residual 或 per-sample fitting。

| Profile | Result |
|---|---:|
| core | `3/3` pass, edge mean/max `1.7249 / 1.8926 px` |
| review | `7/7` pass, edge mean/max `1.6379 / 2.7155 px` |
| broad | `18/18` pass, edge mean/max `2.2408 / 2.9714 px`, improved/regressed `8 / 0`, `22` candidates/sample |

Phase29 v9 解决了之前 broad pressure set 中最难的 `050`、`110`、`123`，并保持 `187` 低于 `3 px`。需要诚实说明的是：v9 是 support-gated registration/evidence method。对于反光、背景异常、弱目标、双边缘等困难场景，它会抑制不可靠的背景热边缘，输出目标相关的 LWIR 注册证据，而不是声称完整重建全场景热图。

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
| 文档索引 | `docs/README.md` |
| 阶段计划 | `task_plan.md` |
| 技术发现 | `findings.md` |
| 执行记录 | `progress.md` |

## 上传目录总览

| 文件夹 | 主要内容 |
|---|---|
| `calibration/` | 用户相机标定文件，包括 RGB/LWIR/UV/depth 相关内参、畸变、外参和设备标定 JSON。 |
| `darklight_mm5/` | MM5 RGB/LWIR/depth 配准、融合、评估主工作区；当前验收版本在 `calibration_only_method/phase29/`。 |
| `docs/` | 项目文档索引、设计记录、重命名清单、整理脚本。 |
| `mar_scholar_compare/` | Scene 282 和 MAR 论文式对比、历史图表和展示材料。 |
| `mm5_calib_benchmark/` | MM5 多方法 benchmark 框架和 split index；清理后只保留当前需要的 split 索引输出。 |
| `mm5_ivf/` | MM5 历史实验代码，包含数据构建、训练、融合和可视化探索。 |
| `peizhun_jiguang/` | DA1501A 单点激光测距辅助 FPGA/HLS IP 原型。 |
| `runs/` | 历史运行结果、人工分析图、Word 报告和阶段性输出。 |

本地环境目录如 `.venv/`、`.git_ssh/`、HLS build/cache、Python `__pycache__` 不作为项目内容同步。

## 项目目标

本仓库围绕 MM5 多模态数据集，保留了一条从离线标定配准到 FPGA 实时化的完整工作链路：

1. 在 MM5 数据集上，只使用用户标定数据、raw RGB/LWIR/depth 图像和棋盘格采集来生成配准结果。
2. 保持 MM5 official aligned 图像只用于评估，不用于反推参数、teacher residual 或 per-sample fitting。
3. 将离线几何成果逐步转化为可综合的 HLS IP，用 DA1501A 单点激光测距和 range-bin LUT 替代 dense depth runtime。
4. 将最终可验收结果收敛到 Phase29 v9，并把旧版本结果压缩为文字记忆，避免工作区继续膨胀。

## 方法边界

当前 Phase29 v9 生成阶段允许使用：

- `calibration/` 中的用户标定文件；
- MM5 index 指向的 raw RGB1、raw LWIR16、raw depth 图像；
- 原始 calibration-board captures；
- calibration-derived board correspondences；
- depth foreground/support masks；
- 固定几何模型、固定输出 canvas、固定 crop/affine 参数；
- raw/depth-only selector gates。

当前 Phase29 v9 生成阶段不允许使用：

- MM5 aligned RGB/T16 作为参数来源；
- official aligned transform、aligned template、teacher residual；
- 读取 aligned 图像后针对单张样本做 per-sample fitting 或调参；
- 用 aligned 指标选择最终生成候选。

aligned 图像只在评估阶段读取，用于报告 NCC、edge distance、heatmap、oracle ceiling 和视觉验收对比。

## 版本记忆

旧 README 中的 Phase25 和 FPGA 说明仍然是项目历史的重要部分；当前 README 将它们整理为“版本记忆”，避免再把旧输出目录当作当前验收对象。

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
单点激光距离 -> range-bin LUT -> 固定点 LWIR warp -> RGB/LWIR fusion
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

## 复现当前 Phase29 v9

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

## 仓库结构

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

### `calibration/`

保存项目使用的用户标定文件，是 MM5 配准、benchmark 和 FPGA 参数导出的共同基础。

主要文件包括：

- `def_stereocalib_THERM.yml`：RGB/LWIR stereo calibration。
- `def_thermalcam_ori.yml`：raw thermal camera intrinsics。
- `def_stereocalib_UV.yml`、`def_uvcam_ori.yml`：UV 方向相关标定。
- `def_stereocalib_cam.yml`：RGB/depth 或相机组相关标定。
- `calib_device_0.json`、`calib_device_1.json`：设备级标定信息。

### `darklight_mm5/`

MM5 RGB/LWIR/depth 配准与融合主工作区。当前核心是 `calibration_only_method/phase29/`。Phase29 需要的旧 Phase28 helper 已整合到 `phase29/phase29_integrated_helpers.py`，不再保留独立 `phase28/` 目录。

### `mm5_calib_benchmark/`

完整的 MM5 多方法 benchmark 框架仍保留代码和配置。当前 active output 只保留 split index：

```text
mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv
```

这个文件是样本路径配对索引，不是普通生成结果，Phase29 仍会使用它。

### `peizhun_jiguang/`

FPGA 实时化方向的核心目录。它把离线几何结果转成 DA1501A 单点激光测距辅助的 HLS IP。

```text
peizhun_jiguang/
|- config/
|- docs/
|- generated/
|- hls/
`- scripts/
```

### `docs/`、`runs/`、`mar_scholar_compare/`、`mm5_ivf/`

- `docs/`：项目文档索引、清单和整理工具。
- `runs/`：阶段性运行结果、人工分析材料和 Word 报告。
- `mar_scholar_compare/`：Scene 282、MAR 论文式比较、图表与展示材料。
- `mm5_ivf/`：MM5 历史数据构建、训练、融合和可视化探索代码。

## 数据与环境说明

仓库保存代码、标定、关键输出和分析材料；完整 MM5 原始数据通常仍位于本机数据盘。重新运行 Phase29 时，需要 index 指向的 raw RGB、raw LWIR、raw depth 和 evaluation-only aligned 图像实际存在。

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

## 当前一句话总结

这个仓库现在保留了一条清晰主线：以 Phase29 v9 作为 MM5 raw/calibration/depth-only 的当前验收配准方案，同时保留 DA1501A 单点激光测距辅助 FPGA/HLS 原型作为实时化方向。旧版本的指标和经验已经整合进文档，生成物则从 active workspace 中清理，便于后续验收、复现和 GitHub 同步。
