# Phase 25 Depth-Assisted LWIR Refinement

## Constraint
RGB stays fixed to the Phase 21 calibration-derived canvas. The main LWIR geometry stays the Phase 24 calibration-board affine result. Depth is used only from the raw per-sample depth image and stereo calibration; MM5 aligned images are evaluation-only.

## Depth Branch
- `phase25_depth_registered_global_shift`: selects one global LWIR residual translation by minimizing raw depth foreground-boundary distance to generated LWIR edges.
- `phase25_depth_registered_global_shift_depth_fill`: applies that depth-selected registration and fills only the new invalid border from the dense depth projection.
- `phase25_depth_project_raw_lwir`: samples raw LWIR through RGB depth and RGB->LWIR calibration as a pure dense depth-registration diagnostic.
- `phase25_depth_holefill_keep_phase24_valid`: old conservative depth-fill diagnostic, retained only for comparison.

## Bridge Target
- retained bridge LWIR NCC mean: `0.9233`
- retained bridge LWIR NCC min: `0.9064`

## Calibration Geometry
- checkerboard correspondence points: `1848`
- best board transform: `affine_lmeds`, board RMSE `7.8255px`

## Phase 24 Baseline
- LWIR NCC mean/min: `0.9182` / `0.9118`
- RGB NCC mean/min: `0.9865` / `0.9724`

## Promoted Phase 25 Candidate
- candidate: `phase25_depth_registered_global_shift_depth_fill`
- LWIR NCC mean/min: `0.9321` / `0.9261`
- RGB NCC mean/min: `0.9865` / `0.9724`
- depth-selected residual shift: `dx=-2`, `dy=2`
- depth boundary score gain: `1.1652698516845703`
- valid policy: `shifted_phase24_valid_mask`

## Depth Registration Without Fill
- LWIR NCC mean/min: `0.9237` / `0.9174`

## Legacy Depth Fill Diagnostic
- LWIR NCC mean/min: `0.9232` / `0.9163`

## Depth Valid-Union Diagnostic
- LWIR NCC mean/min: `0.9206` / `0.9131`

## Depth-Only Diagnostic
- LWIR NCC mean/min: `0.7638` / `0.7441`

## Best Evaluated Candidate
- candidate: `phase25_depth_registered_global_shift_depth_fill`
- LWIR NCC mean/min: `0.9321` / `0.9261`

## Candidate Table

| candidate | method | LWIR NCC mean | LWIR NCC min | edge mean | valid mean |
|---|---|---:|---:|---:|---:|
| phase25_depth_registered_global_shift_depth_fill | depth_boundary_registered_global_translation_depth_fill | 0.9321 | 0.9261 | 14.8499 | 0.9315 |
| phase25_depth_registered_global_shift | depth_boundary_registered_global_translation | 0.9237 | 0.9174 | 16.9251 | 0.9315 |
| phase25_depth_holefill_keep_phase24_valid | phase24_affine_depth_border_holefill | 0.9232 | 0.9163 | 16.9820 | 0.9348 |
| phase25_median_holefill_control | phase24_affine_median_holefill_control | 0.9209 | 0.9144 | 18.2294 | 0.9348 |
| phase25_depth_holefill_union_valid | phase24_affine_depth_border_holefill_union_valid | 0.9206 | 0.9131 | 20.5837 | 0.9952 |
| phase24_affine_lmeds_baseline | phase24_affine_lwir_only | 0.9182 | 0.9118 | 15.2661 | 0.9348 |
| phase25_depth_project_raw_lwir | depth_project_raw_lwir | 0.7638 | 0.7441 | 20.8219 | 0.8936 |
