# 指标说明

- 本说明按 thermal 图像尺寸 640 x 512 = 327680 像素来举例。
- `pixel_accuracy` 的含义是：在所有有效评测像素里，预测类别与目标 GT 完全一致的像素比例。
- 如果 `pixel_accuracy = 0.99`，表示大约 99% 的有效像素分类是正确的，约 1% 是错误的。
- 如果整张图都进入评测，1% 的错误大约对应 3277 个像素。
- 但 `pixel_accuracy` 不是一个直接的几何位移像素量，它并不等价于 `0.01 px` 的校准偏差。
- 这次新增的 `overall_region_error_px` 更偏向整体区域误差：它对整个前景区域做双向平均距离，而不是只看边界。
- 为了让 thermal 与 UV 这两种不同分辨率结果可以直接比较，这次把它进一步归一化为 `normalized_overall_region_error = overall_region_error_px / image_diagonal_px`。
- 这个归一化版本可以理解为“整体区域平均误差占图像对角线的比例”，越低说明整体校准越精确。
- 原来的 `keypoint_transfer_error_px` 仍然保留，但它更偏向边界平均误差。
- `mean_iou` 的含义是：先对每个类别计算 `IoU = intersection / union`，然后对各类取平均。它比 `pixel_accuracy` 更敏感于前景错位、类别混淆和边界问题。
- `boundary_f1` 更强调轮廓是否贴边，它适合观察目标边缘有没有“糊在一起”或者“错开一圈”。
- `valid_warp_coverage` 表示有效投影覆盖率。如果它偏低，说明虽然局部结果可能看起来不错，但实际上只有一部分区域被稳定投影到了目标图像里。
- 如果一个方法的 `normalized_overall_region_error` 很低，但 `valid_warp_coverage` 也很低，通常表示它只在一小块有效区域里对齐得很好，不能单独据此判为整体最优。
- 实际上，一个方法即使 `pixel_accuracy` 很高，也可能因为背景占比大而让 `mean_iou` 仍然一般，这通常意味着前景轮廓和局部区域还没有真正对齐好。

## 当前排序结果（按 normalized_overall_region_error 从低到高，也就是从最好到最差）

- 1. M7: normalized_overall_region_error=0.18%diag, overall_region_error_px=1.46px, mean_iou=0.4505, boundary_f1=0.2609, pixel_accuracy=0.6359, valid_warp_coverage=0.0478, keypoint_transfer_error_px=10.63px
- 2. M5: normalized_overall_region_error=0.49%diag, overall_region_error_px=4.03px, mean_iou=0.4737, boundary_f1=0.3244, pixel_accuracy=0.9682, valid_warp_coverage=0.9279, keypoint_transfer_error_px=13.55px
- 3. M4: normalized_overall_region_error=0.55%diag, overall_region_error_px=4.49px, mean_iou=0.4348, boundary_f1=0.2407, pixel_accuracy=0.9639, valid_warp_coverage=0.9460, keypoint_transfer_error_px=16.48px
- 4. M6: normalized_overall_region_error=0.59%diag, overall_region_error_px=4.85px, mean_iou=0.4291, boundary_f1=0.2339, pixel_accuracy=0.9613, valid_warp_coverage=0.9218, keypoint_transfer_error_px=17.11px
- 5. M2: normalized_overall_region_error=1.43%diag, overall_region_error_px=11.71px, mean_iou=0.2657, boundary_f1=0.2277, pixel_accuracy=0.9643, valid_warp_coverage=0.9481, keypoint_transfer_error_px=16.70px
- 6. M1: normalized_overall_region_error=1.54%diag, overall_region_error_px=12.59px, mean_iou=0.2549, boundary_f1=0.2190, pixel_accuracy=0.9613, valid_warp_coverage=0.9406, keypoint_transfer_error_px=17.54px

## 如何理解 `pixel_accuracy = 0.99`

- 它表示 99% 的有效像素类别判断正确。
- 它并不保证物体轮廓已经完全贴合。
- 如果必须选一个最能代表“整体校准精确度”的主参数，这里更推荐 `normalized_overall_region_error`。
- 因此在看校准质量时，应该结合 `mean_iou`、`boundary_f1`、`normalized_overall_region_error` 和 `keypoint_transfer_error_px` 一起判断。