# MM5_test: MM5 多模态图像实时配准与融合工作区

这个仓库是围绕 MM5 多模态数据集建立的配准、标定、评估和 FPGA 实时化工作区。当前主线目标是：在不把 MM5 aligned 图像用于参数反推的前提下，利用用户自己的标定数据、原始 RGB/LWIR/depth 图像、棋盘格采集和近红外单点激光测距机信息，逐步形成可以部署到 FPGA 的多模态图像实时配准与融合方案。

当前最重要的两个成果是：

1. `darklight_mm5/calibration_only_method` 中的 Phase25 depth-assisted registration。它是目前离线验证效果最好的 MM5 calibration-only 配准方案。
2. `peizhun_jiguang` 中的 DA1501A 近红外单点激光测距机辅助 FPGA 方案。它把 Phase25 的几何思想转成一个可综合的单 HLS IP：`phase25_laser_register_fuse_ip_top`。

## 当前结论

### Phase25 离线配准效果

Phase25 在三张选定暗光样本 `106,104,103` 上保持 RGB 对齐质量不退化，同时提升 LWIR 配准质量。

| 方法 | RGB NCC mean/min | LWIR NCC mean/min | LWIR edge distance mean | 说明 |
|---|---:|---:|---:|---|
| Phase24 baseline | `0.9865 / 0.9724` | `0.9182 / 0.9118` | `15.2661 px` | 棋盘格推导的 LWIR affine |
| Phase25 registration only | `0.9865 / 0.9724` | `0.9237 / 0.9174` | `16.9251 px` | raw depth boundary 选择 residual shift |
| Phase25 promoted | `0.9865 / 0.9724` | `0.9321 / 0.9261` | `14.8499 px` | residual shift + depth border fill |
| retained bridge target | - | `0.9233 / 0.9064` | - | 之前 aligned bridge 水平 |

当前推荐候选：

```text
phase25_depth_registered_global_shift_depth_fill
```

这说明在这三张样本上，Phase25 promoted 已经超过 retained bridge 的 LWIR NCC mean/min，同时 RGB 仍保持 aligned-level 的 `0.9865 / 0.9724`。

### FPGA 激光辅助配准状态

最终 FPGA 方向不是复制 dense depth 图像，而是把 Phase25 的几何结构转成“单点距离测量 + range-bin LUT + 固定点 LWIR warp + RGB/LWIR fusion”的实时方案。

当前 FPGA 目标器件暂定：

```text
xczu15eg
```

当前 HLS 具体 part：

```text
xczu15eg-ffvb1156-2-e
```

最终只保留一个导出的 HLS IP top：

```text
phase25_laser_register_fuse_ip_top
```

这个单 IP 已经合并以下功能：

- DA1501A Protocol 1 接收帧解析；
- checksum、reserved byte、status 检查；
- distance_mm、range age、valid/stale/blind-zone/fallback 状态维护；
- Phase24 fallback geometry；
- Phase25-seeded range-bin 参数选择；
- fixed-point LWIR inverse affine warp；
- packed RGB/LWIR fusion；
- debug 输出：距离、状态 flags、range age。

当前 unified IP HLS synthesis 结果：

| Top | Part | Estimated clock | Fmax | Latency | Resource |
|---|---|---:|---:|---:|---|
| `phase25_laser_register_fuse_ip_top` | `xczu15eg-ffvb1156-2-e` | `7.300 ns` | `136.99 MHz` | `307244-307310 cycles`, about `3.072-3.073 ms` per `640x480` frame | `0 BRAM18K`, `8 DSP`, `4281 FF`, `7302 LUT`, `0 URAM` |

重要限制：DA1501A 是单点测距机，并且资料中存在近距离盲区。当前 MM5 近距离桌面样本不能直接声称已经达到真实 DA1501A 实测 Phase25 性能。现在的 FPGA IP 是 Phase25 seed/fallback + 激光 range-bin 结构，后续还需要真实 laser-to-camera 标定和 range-bin 采样。

## 快速入口

| 目标 | 入口 |
|---|---|
| 读当前最佳配准方法 | [`darklight_mm5/calibration_only_method/README.md`](darklight_mm5/calibration_only_method/README.md) |
| 运行 Phase25 | [`darklight_mm5/calibration_only_method/run_phase25_depth_assisted.py`](darklight_mm5/calibration_only_method/run_phase25_depth_assisted.py) |
| 看 Phase25 报告 | [`darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/reports/dl_p25_report_p25.md`](darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/reports/dl_p25_report_p25.md) |
| 看 Phase25 指标 | [`dl_p25_sum_p25.csv`](darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/metrics/dl_p25_sum_p25.csv) |
| 看 depth residual shift 评分 | [`dl_p25_score_p25.csv`](darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/metrics/dl_p25_score_p25.csv) |
| 读 FPGA 激光辅助方案 | [`peizhun_jiguang/README.md`](peizhun_jiguang/README.md) |
| 读 FPGA strategy | [`peizhun_jiguang/docs/fpga_strategy.md`](peizhun_jiguang/docs/fpga_strategy.md) |
| 读 DA1501A 协议整理 | [`peizhun_jiguang/docs/rangefinder_protocol.md`](peizhun_jiguang/docs/rangefinder_protocol.md) |
| 读验证计划 | [`peizhun_jiguang/docs/verification_plan.md`](peizhun_jiguang/docs/verification_plan.md) |
| 读文档目录索引 | [`docs/README.md`](docs/README.md) |
| 追踪任务计划 | [`task_plan.md`](task_plan.md) |
| 追踪实验发现 | [`findings.md`](findings.md) |
| 追踪执行进度 | [`progress.md`](progress.md) |

## 核心原则

Phase25 的生成阶段允许使用：

- `calibration/` 中的相机标定文件；
- MM5 index 指向的 raw RGB1、raw LWIR16、raw depth 图像；
- 原始 calibration-board captures；
- 从原始数据和标定数据计算出的棋盘格角点、board correspondence、depth foreground boundary；
- 固定几何模型、固定输出 canvas、固定 crop/affine 参数。

Phase25 的生成阶段不允许使用：

- MM5 aligned RGB/T16 图像作为配准参数来源；
- official aligned transform、aligned template、teacher residual 等从参考结果反推的参数；
- 针对单张样本读取 aligned 图像后做 per-sample fitting 或调参。

MM5 aligned 图像只在评估阶段读取，用于报告 NCC、edge distance 等指标。这个约束是本仓库区分“复现 aligned 效果”和“偷用 aligned 结果反推参数”的核心边界。

## 仓库结构

```text
MM5_test/
|- calibration/
|- darklight_mm5/
|  |- calibration_only_method/
|  |- docs/
|  |- outputs/
|  |- outputs_calibration_plane*/
|  `- teacher_residual_method/
|- mm5_calib_benchmark/
|- mar_scholar_compare/
|- mm5_ivf/
|- peizhun_jiguang/
|- runs/
|- docs/
|- task_plan.md
|- findings.md
|- progress.md
|- .gitignore
`- README.md
```

## `calibration/`

这里保存项目使用的相机标定文件，是 Phase25、benchmark 和 FPGA 参数导出的共同基础。

典型文件：

- `def_stereocalib_THERM.yml`: RGB/LWIR stereo calibration，包含 `CM1/CM2/R/T/R1/R2/P1/P2` 等参数。
- `def_thermalcam_ori.yml`: raw thermal camera intrinsics，用于 raw LWIR rectification/projection。
- `def_stereocalib_UV.yml`、`def_uvcam_ori.yml`: UV 方向相关标定。
- `def_stereocalib_cam.yml`: RGB/depth 或相机组相关标定。
- `calib_device_0.json`、`calib_device_1.json`: 设备级标定信息。

这些文件属于允许参与生成的标定数据。

## `darklight_mm5/`

这是围绕 MM5 dark/light 样本做配准实验的主要工作区。

### `darklight_mm5/calibration_only_method/`

当前最重要的离线算法目录。它保存只基于标定数据和原始输入的 aligned-style 重建方法。

关键文件：

- `run_phase25_depth_assisted.py`: 当前最佳 Phase25 主入口。
- `README.md`: Phase25 方法细节、指标和命令说明。
- `outputs_phase25_depth_assisted/`: 当前最佳输出。
- `outputs_phase25_depth_assisted/metrics/`: 指标表。
- `outputs_phase25_depth_assisted/panels/`: 可视化对照图。
- `outputs_phase25_depth_assisted/reports/`: 报告。
- `run_phase21_canvas_optimization.py`: RGB canvas 相关 helper。
- `run_phase22_stereo_recalib.py`: stereo recalibration helper。
- `run_phase23_lwir_board_offset.py`: LWIR board offset helper。
- `run_phase24_lwir_board_affine.py`: checkerboard-derived LWIR affine helper。
- `diagnose_aligned_canvas.py`: aligned canvas 诊断工具。
- `run_calibration_only.py`: 早期 calibration-only baseline。

Phase25 逻辑：

1. 保留 Phase21 的 fixed RGB canvas，使 RGB 对 MM5 aligned RGB 的 NCC 保持 `0.9865 / 0.9724`。
2. 复用 Phase23 的 LWIR crop offset `(280,115)`。
3. 复用 Phase24 从 `1848` 个 checkerboard correspondence points 拟合出的 `affine_lmeds` LWIR residual transform。
4. 把 raw depth 裁剪到固定 RGB canvas。
5. 使用 `depth < 1000 mm` 提取近景 foreground boundary。
6. 在 `2 px` 半径内搜索三张图共享的 LWIR residual translation。
7. 用 depth-boundary-to-LWIR-edge distance 选择 `dx=-2 px, dy=+2 px`。
8. 只对 residual shift 后新产生的边界无效区域做 dense raw-depth LWIR projection fill。

Phase25 的 depth 不是只用于补洞，它参与 residual registration 参数选择，因此是真正的 depth-assisted registration。

运行命令：

```powershell
Set-Location 'E:\aa_read_yan\aMAR\MAR_bianyuan'
python .\darklight_mm5\calibration_only_method\run_phase25_depth_assisted.py --aligned-ids 106,104,103
```

核心输出：

- [`dl_p25_sum_p25.csv`](darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/metrics/dl_p25_sum_p25.csv): candidate summary。
- [`dl_p25_sum_p25.json`](darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/metrics/dl_p25_sum_p25.json): summary JSON。
- [`dl_p25_met_p25.csv`](darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/metrics/dl_p25_met_p25.csv): 逐样本、逐 candidate 指标。
- [`dl_p25_score_p25.csv`](darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/metrics/dl_p25_score_p25.csv): depth boundary residual shift 评分。
- [`dl_p25_board_pts.csv`](darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/metrics/dl_p25_board_pts.csv): 棋盘格 correspondence 点。
- [`dl_p25_board_tf.csv`](darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/metrics/dl_p25_board_tf.csv): board-derived transform。
- [`dl_p25_report_p25.md`](darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/reports/dl_p25_report_p25.md): Phase25 报告。

### `darklight_mm5/outputs*`

这些目录保存历史 dark/light、calibration-plane、optimized plane、boundary optimization 等输出。它们已经在 Phase30 中统一短命名，例如：

- `dl_ref_*`: reference/早期 darklight 输出；
- `dl_plane_*`: calibration-plane 输出；
- `dl_opt_*`: optimized calibration-plane 输出；
- `dl_bnd_*`: boundary optimization 输出；
- `dl_tflow_*`、`dl_tsample_*`、`dl_tdiag_*`: teacher residual 输出。

这些目录对分析有价值，但当前最佳主线仍是 `calibration_only_method/outputs_phase25_depth_assisted`。

### `darklight_mm5/teacher_residual_method/`

这是 teacher-guided residual alignment 的历史探索目录。它曾用于理解上界和诊断 residual flow，但它读取 official/reference aligned 结果用于离线 teacher，因此不作为最终 deployment 方法。

当前保留它的原因：

- 作为对比和诊断基线；
- 保存 residual flow 上界；
- 帮助解释 Phase25 为什么必须坚持 calibration/raw-only 边界。

## `peizhun_jiguang/`

这是把 Phase25 推向 FPGA 实时配准和融合的核心目录。

目标变化：

- 离线 Phase25 使用 dense depth boundary 选择 residual shift；
- FPGA 运行时不能依赖 dense depth、OpenCV 搜索或 MM5 aligned 图像；
- 近红外单点激光测距机只提供一个距离值；
- 因此 FPGA 方案使用 range-bin LUT 替代 dense-depth residual search。

目录结构：

```text
peizhun_jiguang/
|- config/
|  `- laser_registration_params.json
|- docs/
|  |- fpga_strategy.md
|  |- rangefinder_protocol.md
|  `- verification_plan.md
|- generated/
|  |- laser_lut.csv
|  |- laser_lut.h
|  `- laser_lut.json
|- hls/
|  |- laser_fusion.hpp
|  |- laser_fusion.cpp
|  |- tb_laser_fusion.cpp
|  |- run_hls.tcl
|  |- run_hls_synth.tcl
|  |- run_vitis_hls.ps1
|  |- run_vitis_hls_synth.ps1
|  `- run_manual_clang_check.ps1
`- scripts/
   `- export_laser_lut.py
```

推荐流程：

```powershell
python .\peizhun_jiguang\scripts\export_laser_lut.py `
  --config .\peizhun_jiguang\config\laser_registration_params.json `
  --output-dir .\peizhun_jiguang\generated
```

手动 C++ 验证：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\peizhun_jiguang\hls\run_manual_clang_check.ps1
```

Vitis HLS synthesis 脚本：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\peizhun_jiguang\hls\run_vitis_hls_synth.ps1
```

当前 HLS 注意事项：

- 当前 shell 中 `vitis_hls` 不在 PATH。
- 已找到 Vitis HLS 2022.1 路径：`F:\Vivado\vivado2022.1\Vitis_HLS\2022.1\bin\vitis_hls.bat`。
- 当前 shell 直接跑 Vitis wrapper 可能遇到 Xilinx bundled MSYS `tee.exe/cat.exe` Win32 error 5。
- 已使用 manual clang check 作为代码级验证 fallback。
- synthesis 已在 `xczu15eg-ffvb1156-2-e` 上通过。

## `mm5_calib_benchmark/`

这是更完整的 MM5 多方法 benchmark 框架，用于比较不同跨模态配准和标定方法。

重要子目录：

- `configs/`: 全局配置和方法配置。
- `methods/`: M0 到 M7 的方法实现，例如 official baseline、Zhang/OpenCV、depth bridge、EPnP、MAR edge refine、depth-guided self calibration。
- `eval/`: geometry、mask、boundary、NCC 等评估指标。
- `viz/`: 可视化工具。
- `scripts/`: split 生成、benchmark 运行、Scene 282 对比图和文档生成入口。
- `outputs/mm5_benchmark/`: benchmark 输出和 Scene 282 对比材料。

常用入口：

```powershell
python -m mm5_calib_benchmark.scripts.make_splits
python -m mm5_calib_benchmark.scripts.run_all_methods
python -m mm5_calib_benchmark.scripts.make_scene_2823_comparison
```

重要保留文件：

```text
mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv
```

这个文件是样本/路径配对索引，不是普通输出 artifact，Phase30 整理时特意保留原名。

## `mar_scholar_compare/`

这里保存 Scene 282 相关的论文式、历史 MAR 对比材料。它更偏分析和展示，不是 Phase25 主运行路径。

用途：

- 对比历史 MAR 结果与当前 benchmark 结果；
- 解释 Scene 282 中不同指标；
- 生成 visual panel、ranking bar、normalized error heatmap 等材料。

## `mm5_ivf/`

这是另一条与 MM5 相关的实验、可视化或融合方向工作区。它不是当前 Phase25 配准主线，但作为历史实验记录保留。

## `runs/`

保存阶段性运行结果、人工分析文件、Word 报告和中间产物。Phase30 后，Word 汇总材料已整理到：

```text
runs/reports/word/
```

## `docs/`

项目级文档、清单和整理工具集中在这里。

重要文件：

- [`docs/README.md`](docs/README.md): 文档目录索引。
- [`docs/manifests/rename_manifest_20260428.csv`](docs/manifests/rename_manifest_20260428.csv): Phase30 artifact 重命名映射。
- [`docs/manifests/document_reorg_manifest_20260428.csv`](docs/manifests/document_reorg_manifest_20260428.csv): 文档整理移动映射。
- [`docs/tools/rename_artifacts_phase30.ps1`](docs/tools/rename_artifacts_phase30.ps1): Phase30 短命名脚本。
- `docs/superpowers/specs/`: 之前阶段的方法设计记录。

Phase30 整理结果：

- artifact rename rows: `3151`；
- document move rows: `23`；
- old paths still present: `0`；
- new paths missing: `0`；
- selected output artifact bad basenames: `0`。

## 根目录规划文件

这三个文件是持续协作时的工作记忆：

- [`task_plan.md`](task_plan.md): 阶段计划、当前目标、决策和状态。
- [`findings.md`](findings.md): 实验发现、指标结论、失败路径和分析。
- [`progress.md`](progress.md): 逐步运行记录、验证命令、清理和同步记录。

继续优化前建议先读这三个文件，尤其是 `findings.md` 中的 Phase25、Phase29、Phase30 记录。

## 数据依赖

仓库中保存了代码、标定、输出和分析材料，但完整 MM5 原始数据通常仍位于本机路径，例如：

```text
D:\a三模数据\MM5_RAW\...
D:\a三模数据\MM5_ALIGNED\...
D:\a三模数据\MM5_CALIBRATION\...
```

重新运行 Phase25 时，需要 index 指向的 raw RGB、raw LWIR、raw depth、aligned evaluation 图像在当前机器上真实存在。

注意：Phase25 会读取 aligned 图像做评估，但不会用 aligned 图像生成或选择配准参数。

## 环境依赖

主要 Python 依赖：

- `numpy`
- `opencv-python`
- `scipy`
- `scikit-image`
- `python-docx`

推荐使用项目根目录已有 `.venv` 或等价 Python 环境。`.venv` 不同步到 GitHub。

FPGA/HLS 相关：

- Vitis HLS 2022.1 已在本机发现；
- 临时目标器件族为 `xczu15eg`；
- 当前综合 full part 为 `xczu15eg-ffvb1156-2-e`；
- HLS 生成工程目录和本地 build 产物不纳入 GitHub 同步。

## 推荐复现流程

### 1. 复现当前最佳 Phase25

```powershell
Set-Location 'E:\aa_read_yan\aMAR\MAR_bianyuan'
python .\darklight_mm5\calibration_only_method\run_phase25_depth_assisted.py --aligned-ids 106,104,103
```

看报告：

```text
darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/reports/dl_p25_report_p25.md
```

看 summary：

```text
darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/metrics/dl_p25_sum_p25.csv
```

看 panels：

```text
darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/panels/
```

### 2. 导出激光 range-bin LUT

```powershell
python .\peizhun_jiguang\scripts\export_laser_lut.py `
  --config .\peizhun_jiguang\config\laser_registration_params.json `
  --output-dir .\peizhun_jiguang\generated
```

### 3. 验证 FPGA HLS C++ 代码级行为

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\peizhun_jiguang\hls\run_manual_clang_check.ps1
```

期望输出：

```text
tb_laser_fusion PASS
```

### 4. HLS synthesis

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\peizhun_jiguang\hls\run_vitis_hls_synth.ps1
```

最终导出 IP top：

```text
phase25_laser_register_fuse_ip_top
```

## 文件命名规则

Phase30 后，输出文件使用短前缀区分来源：

| 前缀 | 含义 |
|---|---|
| `dl_p25` | Phase25 depth-assisted registration |
| `dl_ref` | 早期 darklight/reference 输出 |
| `dl_plane` | calibration-plane 输出 |
| `dl_opt` | optimized calibration-plane 输出 |
| `dl_bnd` | boundary optimization 输出 |
| `dl_tflow` | teacher residual global flow 输出 |
| `dl_tsample` | teacher residual sample-flow upper bound |
| `dl_tdiag` | teacher residual diagnostics |
| `m0th` 到 `m7th` | benchmark thermal 方法 |
| `m0uv` 到 `m5uv` | benchmark UV 方法 |
| `cmp282` | Scene 282 comparison |
| `legacy_eng` | legacy engineered best |
| `legacy_pap` | legacy paper final |
| `run` | runs 目录中的人工/阶段性结果 |
| `laser` | peizhun_jiguang generated LUT |

两个清单可以用于追溯旧名：

- `docs/manifests/rename_manifest_20260428.csv`
- `docs/manifests/document_reorg_manifest_20260428.csv`


## 当前开发路线

1. 保持 Phase25 作为离线 best-known registration baseline。
2. 在真实 DA1501A + RGB/LWIR 机械安装完成后，采集 laser-to-camera 标定数据。
3. 把真实距离分段写入 `peizhun_jiguang/config/laser_registration_params.json`。
4. 重新导出 `laser_lut.*`。
5. 用 `phase25_laser_register_fuse_ip_top` 做 HLS/IP package。
6. 在 FPGA 上接入 UART、VDMA/ping-pong frame buffer、RGB/LWIR frame sync 和 fused output。
7. 用真实硬件采样验证 range-bin 是否能接近 Phase25 离线 dense-depth 水平。

## 当前一句话总结

这个仓库现在包含一条完整链路：从 MM5 raw/calibration-only 的 Phase25 最佳离线配准，到 DA1501A 单点近红外激光测距辅助的 FPGA 单 IP 实时配准与融合原型。Phase25 给出当前效果上界和几何种子，`peizhun_jiguang` 把它转成可综合、可接入实时视频系统的硬件实现路径。
