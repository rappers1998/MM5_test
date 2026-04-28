# Calibration Plane Evaluation

Current version: `calibration_plane`.

## Mean Metrics

| Metric | Value |
|---|---:|
| target_ncc | 0.2599 |
| target_mi | 0.5342 |
| target_edge_distance | 5.6309 |
| target_valid_ratio | 0.8324 |
| raw_rgb_ncc | 0.2403 |
| raw_rgb_edge_distance | 2.6926 |
| valid_ratio | 0.8324 |
| fusion_entropy_gain_vs_raw_rgb | 1.3861 |
| fusion_alpha_coverage | 0.0088 |

## Per Sample

| sample | target_ncc | target_edge_distance | raw_rgb_ncc | raw_rgb_edge_distance | valid_ratio |
|---|---:|---:|---:|---:|---:|
| 103_seq385 | 0.3679 | 4.0552 | 0.3098 | 3.5582 | 0.8324 |
| 104_seq386 | 0.1613 | 6.0891 | 0.1519 | 3.0997 | 0.8324 |
| 106_seq388 | 0.2506 | 6.7486 | 0.2592 | 1.4199 | 0.8324 |

## Visual Panels

- `darklight_mm5\outputs_calibration_plane\evaluation_panels\dl_plane_s103_eval_panel.png`
- `darklight_mm5\outputs_calibration_plane\evaluation_panels\dl_plane_s104_eval_panel.png`
- `darklight_mm5\outputs_calibration_plane\evaluation_panels\dl_plane_s106_eval_panel.png`
