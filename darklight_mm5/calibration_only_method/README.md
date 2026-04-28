# Phase25 Depth-Assisted MM5 Registration

This folder keeps the latest calibration-derived registration path for the three
selected dark MM5 samples: `106`, `104`, and `103`.

## Rule

Generation uses only:

- MM5 calibration files under `calibration/`
- original calibration-board captures referenced by the MM5 index
- raw RGB1, raw LWIR16, and raw depth images

MM5 aligned RGB/T16 images are used only after generation for evaluation. They
are not used to fit, tune, select, or refine the registration parameters.

## Current Best Method

Run:

```powershell
python .\darklight_mm5\calibration_only_method\run_phase25_depth_assisted.py --aligned-ids 106,104,103
```

Output:

```text
darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted/
```

The promoted candidate is:

```text
phase25_depth_registered_global_shift_depth_fill
```

Pipeline:

1. Keep RGB fixed to the Phase21 calibration-derived canvas.
2. Use the Phase23 LWIR crop offset `(280,115)`.
3. Use the Phase24 checkerboard-derived `affine_lmeds` LWIR transform.
4. Crop raw depth to the fixed RGB canvas.
5. Extract the near foreground boundary with `depth < 1000 mm`.
6. Select one shared residual LWIR translation by minimizing depth-boundary
   distance to generated LWIR edges.
7. Apply the selected residual shift `dx=-2 px`, `dy=+2 px`.
8. Fill only the new invalid border from the dense raw-depth LWIR projection.

## Result On The Three Test Images

| Candidate | RGB NCC mean/min | LWIR NCC mean/min | LWIR edge distance mean |
|---|---:|---:|---:|
| Phase24 baseline | `0.9865 / 0.9724` | `0.9182 / 0.9118` | `15.2661 px` |
| Phase25 depth registration only | `0.9865 / 0.9724` | `0.9237 / 0.9174` | `16.9251 px` |
| Phase25 promoted | `0.9865 / 0.9724` | `0.9321 / 0.9261` | `14.8499 px` |

The retained bridge target on these samples is about:

```text
LWIR NCC mean/min = 0.9233 / 0.9064
```

Phase25 exceeds that retained bridge target on the three selected samples while
keeping the MM5 aligned images evaluation-only.

## Important Files

- `run_phase25_depth_assisted.py` - latest depth-assisted registration entry.
- `outputs_phase25_depth_assisted/phase25_depth_assisted_report.md` - run report.
- `outputs_phase25_depth_assisted/metrics/phase25_depth_assisted_summary.csv` -
  candidate summary.
- `outputs_phase25_depth_assisted/metrics/phase25_depth_registration_scores.csv` -
  depth-boundary registration scores.
- `outputs_phase25_depth_assisted/panels/` - visual comparison panels.

Older Phase output folders were intentionally excluded from the GitHub sync. The
helper scripts from Phase21-Phase24 are still kept because Phase25 reuses their
stable calibration, checkerboard, and metric utilities.
