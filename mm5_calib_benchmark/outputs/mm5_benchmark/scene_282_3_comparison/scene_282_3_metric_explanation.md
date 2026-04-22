# 指标说明

- 本说明按 thermal 图像尺寸 640 x 512 = 327680 像素来举例。
- `pixel_accuracy` 的含义是：在所有有效评测像素里，预测类别与目标 GT 完全一致的像素比例。
- 如果 `pixel_accuracy = 0.99`，表示大约 99% 的有效像素分类是正确的，约 1% 是错误的。
- 如果整张图都进入评测，1% 的错误大约对应 3277 个像素。
- 但 `pixel_accuracy` 不是一个直接的几何位移像素量，它并不等价于 `0.01 px` 的校准偏差。
- 在这个 benchmark 里，更接近几何校准像素误差的量是 `keypoint_transfer_error_px`，单位就是像素，表示校准后关键点迁移的平均偏差。
- `mean_iou` 的含义是：先对每个类别计算 `IoU = intersection / union`，然后对各类取平均。它比 `pixel_accuracy` 更敏感于前景错位、类别混淆和边界问题。
- 实际上，一个方法即使 `pixel_accuracy` 很高，也可能因为背景占比大而让 `mean_iou` 仍然一般，这通常意味着前景轮廓和局部区域还没有真正对齐好。

## 当前排序结果（按 test mean_iou 从高到低）

- 1. M7: mean_iou=0.5797, pixel_accuracy=0.9691, keypoint_transfer_error_px=6.75px
- 2. M5: mean_iou=0.4731, pixel_accuracy=0.9686, keypoint_transfer_error_px=13.71px
- 3. M4: mean_iou=0.4302, pixel_accuracy=0.9646, keypoint_transfer_error_px=16.16px
- 4. M6: mean_iou=0.4178, pixel_accuracy=0.9647, keypoint_transfer_error_px=15.09px
- 5. M2: mean_iou=0.2649, pixel_accuracy=0.9642, keypoint_transfer_error_px=16.75px
- 6. M1: mean_iou=0.2569, pixel_accuracy=0.9618, keypoint_transfer_error_px=17.48px

## 如何理解 `pixel_accuracy = 0.99`

- 它表示 99% 的有效像素类别判断正确。
- 它并不保证物体轮廓已经完全贴合。
- 因此在看校准质量时，应该结合 `mean_iou`、`boundary_f1` 和 `keypoint_transfer_error_px` 一起判断。