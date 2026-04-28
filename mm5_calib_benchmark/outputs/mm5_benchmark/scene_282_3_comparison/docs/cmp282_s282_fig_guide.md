# Scene 282 图像阅读说明

- 图内标题统一改成英文，是因为当前 OpenCV 文本绘制不支持稳定的中文渲染；详细中文解释以本文件和 `README.md` 为准。

## 推荐阅读顺序

1. 先看 `scene_282_3_all_methods_comparison.png`，快速理解每个方法在同一场景上的视觉差异。
2. 再看 `scene_282_3_thermal_method_ranking_bar.png`，确认 test 集上的整体排名，不只看单张图。
3. 然后看 `scene_282_3_thermal_per_scene_normalized_error.png`，确认每一张 test 图片上到底是哪种方法更稳。
4. 最后看 `scene_282_3_mar_history_panel.png` 和 `scene_282_3_mar_history_note.md`，把新 benchmark 和历史 MAR 全流程结果对应起来。

## 核心图像说明

### `scene_282_3_all_methods_comparison.png`

- 显示效果：第一行给出 RGB 源图、thermal 目标图和 thermal GT；下面每个方法各占一行，分别展示 prediction、contours、error 三个视角。
- 重点看什么：`prediction` 看整体类别区域有没有压到正确物体上，`contours` 看边界是否贴边，`error` 看错误主要来自漏检、误检还是类别混淆。
- 重要参数：`mean_iou`、`boundary_f1`、`pixel_accuracy`、`valid_warp_coverage`、`overall_region_error_px`。
- 颜色含义：误差图里绿色表示预测正确，蓝色表示误检前景，橙色表示漏掉 GT 前景，紫色表示前景类别错分，黑色表示无效投影区域。

### `scene_282_3_thermal_method_ranking_bar.png` / `scene_282_3_thermal_overall_region_ranking_bar.png`

- 显示效果：把 Scene 282 主对比里出现的六个 thermal 方法放到同一个 test 集排名图里，分别展示区域误差、归一化区域误差、`mean_iou`、`pixel_accuracy` 和 `keypoint_transfer_error_px`。
- 重点看什么：先看 `normalized_overall_region_error` 和 `overall_region_error_px` 判断整体校准精度，再结合 `mean_iou` 与 `pixel_accuracy` 判断语义是否也同步改善。
- 重要参数：`normalized_overall_region_error`、`overall_region_error_px`、`mean_iou`、`pixel_accuracy`、`keypoint_transfer_error_px`。

### `scene_282_3_thermal_per_scene_normalized_error.png`

- 显示效果：每一行对应一张 test 图片，每一列对应一个 thermal 方法，单元格内给出该图上的 `normalized_overall_region_error`，白色边框标出该图的最优方法。
- 重点看什么：它可以回答“某个方法是不是靠少数场景拉高平均分”，也能直接看出哪些图片是难例。
- 重要参数：`normalized_overall_region_error`、`best_method`。

### `scene_282_3_mar_history_panel.png`

- 显示效果：把历史 MAR 工程版和论文版的 raw/aligned 结果并排放在一起，同时列出几何诊断信息。
- 重点看什么：对比历史流程在“原始 GT”与“对齐后 GT”下的表现差异，以及几何基线是否成立。
- 重要参数：`pixel_accuracy`、`mean_iou`、`binary_iou_foreground`、`geometry_baseline_ok`、`real_projected_ratio_on_redistort`。

### `scene_282_3_metric_explanation.md` / `scene_282_3_thermal_per_scene_normalized_error_note.md` / `scene_282_3_mar_history_note.md`

- 显示效果：这三份说明文件分别负责解释指标含义、逐图对比结果、以及历史 MAR 结果来源。
- 重点看什么：当图像里已经看到明显好坏差异时，用这三份说明确认“到底应该信哪个指标、每个指标讲的是什么”。
- 重要参数：取决于对应说明文件，但建议优先看 `normalized_overall_region_error`、`mean_iou`、`boundary_f1` 和 `valid_warp_coverage`。

## 当前简要结论

- 当前按 `normalized_overall_region_error` 排名的热成像最优方法是 `M7`：`normalized_overall_region_error=0.18%diag`，`overall_region_error_px=1.46px`，`mean_iou=0.4505`。
- 在当前实现里，`normalized_overall_region_error` 更适合作为跨方法、跨模态的一号主指标；`pixel_accuracy` 可以保留，但不建议单独使用。
- 如果某个方法的 `normalized_overall_region_error` 排名很前，但 `valid_warp_coverage` 很低，需要警惕它是不是只在很小一块区域上对齐得很好。
- Scene 282 的主图和逐图误差表都只聚焦 thermal 方法，因此 `M3` 这类 UV-only 方法不在这些汇总图中；它们仍保留在各自的 per-method benchmark 输出目录里。