# MM5 多模态图像配准与 FPGA 实时融合项目

## 上传目录总览

| 文件夹 | 主要存放内容 |
|---|---|
| `calibration/` | 用户自己的相机标定文件，包括 RGB/LWIR/UV/depth 相关的内参、畸变、外参和设备标定 JSON。 |
| `darklight_mm5/` | MM5 暗光 RGB1 与 LWIR 配准、融合、评估的主工作区；当前最重要结果在 `calibration_only_method/outputs_phase25_depth_assisted/`。 |
| `docs/` | 项目文档索引、流程整理、重命名清单、文件整理清单和辅助脚本。 |
| `mar_scholar_compare/` | Scene 282 和 MAR 论文式对比、历史结果、图表和展示材料。 |
| `mm5_calib_benchmark/` | MM5 多方法配准 benchmark 框架，包括方法实现、配置、评估指标、可视化和 benchmark 输出。 |
| `mm5_ivf/` | MM5 相关的历史实验代码，主要用于数据构建、训练、融合和可视化探索。 |
| `peizhun_jiguang/` | 将 Phase25 配准思想转成 DA1501A 单点激光测距辅助 FPGA/HLS IP 的工程目录。 |
| `runs/` | 阶段性运行结果、人工分析图、Word 报告和历史实验输出。 |

> `.git_ssh/`、`.venv/`、HLS 本地 build/cache 和 Python 缓存属于本地环境文件，不作为 GitHub 项目内容说明。

## 项目目标

本仓库围绕 MM5 多模态数据集，整理了从离线标定配准到 FPGA 实时化的完整工作链路：

1. 在 MM5 数据集上，只使用用户标定数据、raw RGB/LWIR/depth 图像和棋盘格采集来生成配准结果。
2. 保持 MM5 official aligned 图像只用于评估，不用于反推参数、teacher residual 或 per-sample fitting。
3. 将当前最好的 Phase25 配准思路转化为可综合的 HLS IP 核，用 DA1501A 单点激光测距和 range-bin LUT 代替 dense depth runtime。

当前最重要的两条主线是：

- `darklight_mm5/calibration_only_method/`：Phase25 depth-assisted calibration-only registration。
- `peizhun_jiguang/`：Phase25 + DA1501A 激光测距辅助 FPGA 单 IP 原型。

## 当前核心结果

### Phase25 离线配准

Phase25 在三张选定暗光样本 `106,104,103` 上，保持 RGB 对齐质量不退化，同时提升 LWIR 配准质量。

| 方法 | RGB NCC mean/min | LWIR NCC mean/min | LWIR edge distance mean | 说明 |
|---|---:|---:|---:|---|
| Phase24 board-affine baseline | `0.9865 / 0.9724` | `0.9182 / 0.9118` | `15.2661 px` | 棋盘格推导的 LWIR affine baseline。 |
| Phase25 registration only | `0.9865 / 0.9724` | `0.9237 / 0.9174` | `16.9251 px` | raw depth boundary 选择 residual shift。 |
| Phase25 promoted | `0.9865 / 0.9724` | `0.9321 / 0.9261` | `14.8499 px` | residual shift + depth border fill。 |
| retained bridge target | - | `0.9233 / 0.9064` | - | 原 aligned bridge 参考水平。 |

当前推荐候选：

```text
phase25_depth_registered_global_shift_depth_fill
```

核心含义：Phase25 在三张验证样本上已经超过 retained bridge 的 LWIR NCC mean/min，同时 RGB 仍保持接近 aligned-level 的 `0.9865 / 0.9724`。

### FPGA / HLS IP

FPGA 方向不是直接复制 dense depth 图像流程，而是把 Phase25 的几何结果转换成：

```text
单点激光距离 -> range-bin LUT -> 固定点 LWIR warp -> RGB/LWIR fusion
```

最终只保留一个可导出的 HLS IP top：

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

当前 HLS synthesis 参考结果：

| Top | Part | Estimated clock | Fmax | Latency | Resources |
|---|---|---:|---:|---:|---|
| `phase25_laser_register_fuse_ip_top` | `xczu15eg-ffvb1156-2-e` | `7.300 ns` | `136.99 MHz` | `307244-307310 cycles`, about `3.072-3.073 ms` per `640x480` frame | `0 BRAM18K`, `8 DSP`, `4281 FF`, `7302 LUT`, `0 URAM` |

注意：当前 LUT 仍是 Phase25 seed/fallback 原型。真实硬件精度需要完成 DA1501A 与 RGB/LWIR 相机的机械安装、laser-to-camera 标定和 range-bin 实测。

## 快速入口

| 目标 | 路径 |
|---|---|
| 阅读当前最佳配准方法 | [`darklight_mm5/calibration_only_method/README.md`](darklight_mm5/calibration_only_method/README.md) |
| 运行 Phase25 | [`darklight_mm5/calibration_only_method/run_phase25_depth_assisted.py`](darklight_mm5/calibration_only_method/run_phase25_depth_assisted.py) |
| 查看 Phase25 报告 | [`darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/reports/dl_p25_report_p25.md`](darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/reports/dl_p25_report_p25.md) |
| 查看 Phase25 summary 指标 | [`darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/metrics/dl_p25_sum_p25.csv`](darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/metrics/dl_p25_sum_p25.csv) |
| 阅读 FPGA 激光辅助方案 | [`peizhun_jiguang/README.md`](peizhun_jiguang/README.md) |
| 阅读 FPGA strategy | [`peizhun_jiguang/docs/fpga_strategy.md`](peizhun_jiguang/docs/fpga_strategy.md) |
| 阅读 DA1501A 协议整理 | [`peizhun_jiguang/docs/rangefinder_protocol.md`](peizhun_jiguang/docs/rangefinder_protocol.md) |
| 阅读验证计划 | [`peizhun_jiguang/docs/verification_plan.md`](peizhun_jiguang/docs/verification_plan.md) |
| 阅读文档索引 | [`docs/README.md`](docs/README.md) |
| 查看阶段计划 | [`task_plan.md`](task_plan.md) |
| 查看实验发现 | [`findings.md`](findings.md) |
| 查看执行记录 | [`progress.md`](progress.md) |

## 方法边界

Phase25 生成阶段允许使用：

- `calibration/` 中的用户标定文件；
- MM5 index 指向的 raw RGB1、raw LWIR16、raw depth 图像；
- 原始 calibration-board captures；
- 从原始数据和标定数据计算得到的棋盘格角点、board correspondence、depth foreground boundary；
- 固定几何模型、固定输出 canvas、固定 crop/affine 参数。

Phase25 生成阶段不允许使用：

- MM5 aligned RGB/T16 图像作为配准参数来源；
- official aligned transform、aligned template、teacher residual 等从参考结果反推的参数；
- 针对单张样本读取 aligned 图像后做 per-sample fitting 或调参。

MM5 aligned 图像只在评估阶段读取，用于报告 NCC、edge distance 等指标。

## 仓库结构说明

```text
MM5_test/
|- calibration/
|- darklight_mm5/
|  |- calibration_only_method/
|  |- docs/
|  |- outputs*/
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

MM5 暗光 RGB/LWIR 配准与融合主工作区。当前最重要的是 `calibration_only_method/`。

`darklight_mm5/calibration_only_method/` 保存 calibration-only 路线的 Phase21 到 Phase25 脚本，其中当前最佳入口是：

```powershell
python .\darklight_mm5\calibration_only_method\run_phase25_depth_assisted.py --aligned-ids 106,104,103
```

关键输出：

- `outputs_phase25_depth_assisted/metrics/`：指标 CSV/JSON。
- `outputs_phase25_depth_assisted/panels/`：三张样本的可视化对照图。
- `outputs_phase25_depth_assisted/reports/`：Phase25 报告。

`darklight_mm5/teacher_residual_method/` 是 teacher-guided residual 的诊断目录。它用于理解上界和 residual flow，不作为最终部署方法。

### `peizhun_jiguang/`

FPGA 实时化方向的核心目录。它把 Phase25 的离线几何结果转成 DA1501A 单点激光测距辅助的 HLS IP。

目录结构：

```text
peizhun_jiguang/
|- config/
|  `- laser_registration_params.json
|- docs/
|- generated/
|- hls/
`- scripts/
```

常用命令：

```powershell
python .\peizhun_jiguang\scripts\export_laser_lut.py `
  --config .\peizhun_jiguang\config\laser_registration_params.json `
  --output-dir .\peizhun_jiguang\generated
```

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\peizhun_jiguang\hls\run_manual_clang_check.ps1
```

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\peizhun_jiguang\hls\run_vitis_hls_synth.ps1
```

### `mm5_calib_benchmark/`

完整的 MM5 多方法 benchmark 框架，用于比较 official baseline、Zhang/OpenCV、depth bridge、EPnP、MAR edge refine、depth-guided self calibration 等方法。

重要文件：

```text
mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv
```

这个文件是样本路径配对索引，不是普通输出 artifact。

### `docs/`

项目级文档、清单和整理工具。

- `docs/README.md`：文档索引。
- `docs/manifests/rename_manifest_20260428.csv`：Phase30 artifact 重命名映射。
- `docs/manifests/document_reorg_manifest_20260428.csv`：文档整理移动映射。
- `docs/tools/rename_artifacts_phase30.ps1`：Phase30 短命名脚本。
- `docs/superpowers/specs/`：阶段性设计记录。

### `runs/`

保存阶段性运行结果、人工分析材料和 Word 报告。Phase30 后 Word 报告集中在：

```text
runs/reports/word/
```

### `mar_scholar_compare/` 和 `mm5_ivf/`

这两个目录保留历史实验和分析材料：

- `mar_scholar_compare/`：Scene 282、MAR 论文式比较、图表与展示材料。
- `mm5_ivf/`：MM5 相关数据构建、训练、融合和可视化探索。

## 复现流程

### 1. 运行当前最佳 Phase25

```powershell
Set-Location 'E:\aa_read_yan\aMAR\MAR_bianyuan'
python .\darklight_mm5\calibration_only_method\run_phase25_depth_assisted.py --aligned-ids 106,104,103
```

查看输出：

```text
darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/
```

### 2. 导出 FPGA range-bin LUT

```powershell
python .\peizhun_jiguang\scripts\export_laser_lut.py `
  --config .\peizhun_jiguang\config\laser_registration_params.json `
  --output-dir .\peizhun_jiguang\generated
```

### 3. 验证 HLS C++ 行为

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\peizhun_jiguang\hls\run_manual_clang_check.ps1
```

期望输出：

```text
tb_laser_fusion PASS
```

### 4. 运行 HLS synthesis

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\peizhun_jiguang\hls\run_vitis_hls_synth.ps1
```

最终导出 IP top：

```text
phase25_laser_register_fuse_ip_top
```

## 数据与环境说明

仓库保存代码、标定、输出和分析材料；完整 MM5 原始数据通常仍位于本机数据盘。重新运行 Phase25 时，需要 index 指向的 raw RGB、raw LWIR、raw depth 和 evaluation-only aligned 图像实际存在。

主要 Python 依赖：

- `numpy`
- `opencv-python`
- `scipy`
- `scikit-image`
- `python-docx`

FPGA/HLS 相关：

- 已在本机发现 Vitis HLS 2022.1。
- 临时目标器件族为 `xczu15eg`。
- 当前综合 full part 为 `xczu15eg-ffvb1156-2-e`。
- HLS 本地生成工程和 build 产物不纳入 GitHub 同步。

## GitHub 同步命令

只同步这次 README 修改：

```powershell
Set-Location 'E:\aa_read_yan\aMAR\MAR_bianyuan'
git status
git add README.md
git commit -m "Update README project overview"
git push origin main
```

同步当前仓库全部已确认变更：

```powershell
Set-Location 'E:\aa_read_yan\aMAR\MAR_bianyuan'
git status
git add .
git commit -m "Sync MM5 registration and FPGA IP project"
git push origin main
```

当前远程仓库：

```text
https://github.com/rappers1998/MM5_test.git
```

## 当前一句话总结

这个仓库现在包含一条完整链路：从 MM5 raw/calibration-only 的 Phase25 离线配准，到 DA1501A 单点激光测距辅助的 FPGA 单 IP 实时配准与融合原型。Phase25 给出当前离线效果上界和几何种子，`peizhun_jiguang` 将它转成可综合、可接入实时视频系统的硬件实现路径。
