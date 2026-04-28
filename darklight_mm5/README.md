# Dark-Light MM5 RGB1 + LWIR Calibration and Fusion

This is the cleaned current version of the MM5 dark-light RGB1 + LWIR calibration/fusion experiment.
Rejected previous outputs and tuning artifacts have been removed from the active workspace.

## Current Method

Use synchronized frame-1 raw inputs from:

```text
mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv
```

Main input fields:

```text
raw_rgb1_path        -> dark/synchronized visible frame
raw_thermal16_path  -> synchronized LWIR frame
raw_rgb3_path        -> bright visible reference for visualization only
```

The recommended method is the optimized `calibration_plane`:

```text
calibration/def_stereocalib_THERM.yml
raw_rgb1_path
raw_thermal16_path
```

It maps raw LWIR onto the raw RGB1 canvas with a calibrated scene-plane homography for stable visual alignment.

Selected optimized parameters:

```text
lwir_calib_size = 1280x720
plane_depth_mm = 325
t_scale = 1.45
lwir_principal_offset = 14,0
max_residual_shift = 2
fusion_saliency_sigma = 15
fusion_alpha_low = 40
fusion_alpha_high = 96
fusion_alpha_scale = 0.82
fusion_alpha_max = 0.72
fusion_roi_dilate_px = 3
```

Parameter record:

```text
darklight_mm5/calibration_plane_config.json
```

## Run

Run the optimized current version on the three dark samples:

```powershell
python .\darklight_mm5\run_calibration_plane.py --output .\darklight_mm5\outputs_calibration_plane_opt --aligned-ids 106,104,103
```

The original official aligned images are kept only as an evaluation reference under `darklight_mm5/outputs`.
The previous accepted baseline is still kept under `darklight_mm5/outputs_calibration_plane`.

To reproduce the boundary/parameter search:

```powershell
python .\darklight_mm5\optimize_calibration_plane.py --output .\darklight_mm5\outputs_calibration_plane_opt --diagnostic-output .\darklight_mm5\outputs_calibration_plane_boundary
```

## Current Samples

| aligned_id | sequence | split | raw RGB1 mean |
|---:|---:|---|---:|
| 106 | 388 | test | 1.77 |
| 104 | 386 | val | 1.90 |
| 103 | 385 | test | 1.93 |

## Main Outputs

Five-panel review:

```text
darklight_mm5/outputs_calibration_plane_opt/five_panels/*_five_panel.png
```

Each five-panel image contains:

```text
RGB1 Raw dark/synced
RGB3 Raw bright reference
LWIR Raw normalized
LWIR -> RGB1 calibration plane
Fused Result
```

Detailed per-sample outputs:

```text
darklight_mm5/outputs_calibration_plane_opt/samples/<aligned_id>_seq<sequence>/
```

Metrics:

```text
darklight_mm5/outputs_calibration_plane_opt/metrics/dl_opt_met_sample.csv
darklight_mm5/outputs_calibration_plane_opt/metrics/dl_opt_met_reg_stage.csv
darklight_mm5/outputs_calibration_plane_opt/metrics/dl_opt_met_fusion.csv
darklight_mm5/outputs_calibration_plane_opt/dl_opt_eval_sum.json
darklight_mm5/outputs_calibration_plane_opt/dl_opt_eval_ref.csv
darklight_mm5/outputs_calibration_plane_boundary/dl_bnd_opt_resid.csv
darklight_mm5/outputs_calibration_plane_boundary/*_boundary_panel.png
```

## Evaluation Meaning

- `ncc`: normalized cross correlation, higher is better.
- `edge_distance`: mean distance from moving edges to fixed edges, lower is better.
- `valid_ratio`: valid warped area ratio.
- `fusion_entropy_gain_vs_raw_rgb`: display/detail gain from the fusion visualization.
- `target_*`: comparison against the retained `official_reference` result, used only for evaluation.

## Caveat

The fused image is a diagnostic/display fusion. It includes low-light enhancement and thermal saliency injection, so use the registration metrics and valid masks to judge calibration quality separately from visual brightness.
