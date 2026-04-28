# Thermal Per-scene Normalized Error Note

- 这张图逐行展示了 test split 上每一张 thermal 图片的 `normalized_overall_region_error`。
- 白色边框表示该图片上当前误差最低的方法。
- 该指标越低越好，表示整体前景区域误差占图像对角线的比例越小。

## Scene wins

- M1: 1 scenes
- M2: 0 scenes
- M4: 0 scenes
- M5: 1 scenes
- M6: 0 scenes
- M7: 28 scenes

## Per-scene best method

- seq 283: best=M7, normalized_overall_region_error=0.15%diag
- seq 288: best=M7, normalized_overall_region_error=0.15%diag
- seq 293: best=M7, normalized_overall_region_error=0.10%diag
- seq 297: best=M7, normalized_overall_region_error=0.19%diag
- seq 298: best=M7, normalized_overall_region_error=0.19%diag
- seq 299: best=M5, normalized_overall_region_error=0.39%diag
- seq 300: best=M7, normalized_overall_region_error=0.43%diag
- seq 305: best=M7, normalized_overall_region_error=0.19%diag
- seq 306: best=M7, normalized_overall_region_error=0.22%diag
- seq 314: best=M7, normalized_overall_region_error=0.19%diag
- seq 317: best=M7, normalized_overall_region_error=0.12%diag
- seq 318: best=M7, normalized_overall_region_error=0.14%diag
- seq 319: best=M7, normalized_overall_region_error=0.11%diag
- seq 324: best=M7, normalized_overall_region_error=0.21%diag
- seq 325: best=M7, normalized_overall_region_error=0.11%diag
- seq 326: best=M7, normalized_overall_region_error=0.17%diag
- seq 332: best=M7, normalized_overall_region_error=0.22%diag
- seq 340: best=M7, normalized_overall_region_error=0.09%diag
- seq 346: best=M7, normalized_overall_region_error=0.15%diag
- seq 352: best=M7, normalized_overall_region_error=0.11%diag
- seq 354: best=M7, normalized_overall_region_error=0.56%diag
- seq 357: best=M7, normalized_overall_region_error=0.26%diag
- seq 359: best=M7, normalized_overall_region_error=0.19%diag
- seq 380: best=M1, normalized_overall_region_error=0.05%diag
- seq 382: best=M7, normalized_overall_region_error=0.18%diag
- seq 385: best=M7, normalized_overall_region_error=0.13%diag
- seq 388: best=M7, normalized_overall_region_error=0.08%diag
- seq 396: best=M7, normalized_overall_region_error=0.05%diag
- seq 397: best=M7, normalized_overall_region_error=0.07%diag
- seq 400: best=M7, normalized_overall_region_error=0.05%diag