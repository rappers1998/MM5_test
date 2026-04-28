# Phase25 Depth-Assisted MM5 Registration

本目录保存当前效果最好的 MM5 calibration-only 配准方案。它的目标是：只使用用户自己的标定数据、raw RGB/LWIR/depth 图像和原始棋盘格采集，生成尽量接近 MM5 official aligned 水平的 RGB/LWIR 配准结果。

更多项目级说明见根目录 [README](../../README.md)。

## 数据边界

生成阶段允许使用：

- `../../calibration/` 中的 MM5 标定文件；
- MM5 index 指向的 raw RGB1、raw LWIR16、raw depth 图像；
- 原始 calibration-board captures；
- 从原始数据和标定数据计算出的棋盘格角点、board correspondence、depth foreground boundary。

生成阶段不允许使用：

- MM5 aligned RGB/T16 作为参数来源；
- teacher residual、official aligned transform、aligned template 等从参考结果反推的参数；
- 针对单张图读取 aligned 后做 per-sample fitting 或调参。

MM5 aligned 图像只在评估阶段读取，用来报告 NCC、edge distance 等指标。

## 运行入口

在仓库根目录运行：

```powershell
python .\darklight_mm5\calibration_only_method\run_phase25_depth_assisted.py --aligned-ids 106,104,103
```

当前验证样本：

```text
106,104,103
```

输出目录：

```text
darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/
```

最终推荐 candidate：

```text
phase25_depth_registered_global_shift_depth_fill
```

## 方法流程

Phase25 在前几个稳定阶段上继续推进：

1. 保留 `run_phase21_canvas_optimization.py` 得到的 RGB canvas，使 RGB 对 MM5 aligned RGB 的 NCC 保持 `0.9865 / 0.9724`。
2. 复用 `run_phase23_lwir_board_offset.py` 的 LWIR crop offset `(280,115)`。
3. 复用 `run_phase24_lwir_board_affine.py` 从 `1848` 个 checkerboard correspondence points 拟合出的 `affine_lmeds` LWIR residual transform。
4. 在 `run_phase25_depth_assisted.py` 中把 raw depth 裁剪到固定 RGB canvas。
5. 使用 `depth < 1000 mm` 提取近景 foreground boundary。
6. 在 `2 px` 半径内搜索一个三张图共享的 LWIR residual translation。
7. 用 depth-boundary-to-LWIR-edge distance 选择 `dx=-2 px, dy=+2 px`。
8. 应用 residual shift 后，只用 dense raw-depth LWIR projection 填补新产生的边界无效区。

这里的 depth 不是单纯用于补洞；它参与 residual registration 参数选择，所以 Phase25 是真正的 depth-assisted registration。

## 当前三张测试图结果

| Candidate | RGB NCC mean/min | LWIR NCC mean/min | LWIR edge distance mean |
|---|---:|---:|---:|
| Phase24 baseline | `0.9865 / 0.9724` | `0.9182 / 0.9118` | `15.2661 px` |
| Phase25 depth registration only | `0.9865 / 0.9724` | `0.9237 / 0.9174` | `16.9251 px` |
| Phase25 promoted | `0.9865 / 0.9724` | `0.9321 / 0.9261` | `14.8499 px` |

对照 retained bridge target：

```text
LWIR NCC mean/min = 0.9233 / 0.9064
```

Phase25 promoted 在三张测试图上超过 retained bridge target，同时保持 aligned 图像 evaluation-only。

## 输出文件

- [`reports/dl_p25_report_p25.md`](outputs_phase25_depth_assisted/reports/dl_p25_report_p25.md): 运行报告，适合先读。
- [`metrics/dl_p25_sum_p25.csv`](outputs_phase25_depth_assisted/metrics/dl_p25_sum_p25.csv): 候选方法 summary。
- [`metrics/dl_p25_sum_p25.json`](outputs_phase25_depth_assisted/metrics/dl_p25_sum_p25.json): 同一份 summary 的 JSON 版本。
- [`metrics/dl_p25_met_p25.csv`](outputs_phase25_depth_assisted/metrics/dl_p25_met_p25.csv): 逐样本、逐 candidate 指标明细。
- [`metrics/dl_p25_score_p25.csv`](outputs_phase25_depth_assisted/metrics/dl_p25_score_p25.csv): depth boundary 选择 residual shift 的评分。
- [`metrics/dl_p25_board_pts.csv`](outputs_phase25_depth_assisted/metrics/dl_p25_board_pts.csv): 棋盘格 correspondence 点。
- [`metrics/dl_p25_board_tf.csv`](outputs_phase25_depth_assisted/metrics/dl_p25_board_tf.csv): board-derived transform 记录。
- [`panels/`](outputs_phase25_depth_assisted/panels/): 三张样本的可视化对照图。

## 关键脚本

- [`run_phase25_depth_assisted.py`](run_phase25_depth_assisted.py): 当前最佳方法入口。
- [`run_phase24_lwir_board_affine.py`](run_phase24_lwir_board_affine.py): Phase25 复用的 checkerboard-derived LWIR affine helper。
- [`run_phase23_lwir_board_offset.py`](run_phase23_lwir_board_offset.py): LWIR board offset helper。
- [`run_phase22_stereo_recalib.py`](run_phase22_stereo_recalib.py): stereo recalibration helper。
- [`run_phase21_canvas_optimization.py`](run_phase21_canvas_optimization.py): RGB canvas helper。
- [`run_calibration_only.py`](run_calibration_only.py): 早期 calibration-only baseline。
- [`diagnose_aligned_canvas.py`](diagnose_aligned_canvas.py): aligned canvas 诊断脚本。

## 当前结论

Phase25 的重点不是追求大规模全数据集运行，而是在三张指定样本上验证：在 RGB 不退化的前提下，LWIR 可以通过用户标定数据、棋盘格几何和 raw depth boundary 进一步靠近 official aligned 水平。

后续如果继续优化，应优先沿着 Phase25 的 depth-assisted residual registration 做小范围增量验证，而不是回到 teacher/aligned-derived 调参路线。
