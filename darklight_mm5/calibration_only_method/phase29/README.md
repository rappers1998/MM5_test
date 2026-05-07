# Phase29 Strict Broad-Generalization Registration

Phase29 is the current strict calibration/raw/depth registration branch. Generation and selection use calibration files, raw RGB, raw LWIR, raw depth, calibration-board geometry, and raw/depth support masks only. MM5 aligned RGB/T16 are evaluation-only for metrics, heatmaps, oracle ceilings, and visual audit panels.

## Current Version

Phase29 v9 is the current accepted broad-generalization package.

| Profile | Output | Result |
|---|---|---:|
| core | `outputs_core_generalization_v9/` | `3/3` pass, edge mean/max `1.7249 / 1.8926 px`, improved/regressed `2 / 0` |
| review | `outputs_review_generalization_v9/` | `7/7` pass, edge mean/max `1.6379 / 2.7155 px`, improved/regressed `3 / 0` |
| broad | `outputs_broad_generalization_v9/` | `18/18` pass, edge mean/max `2.2408 / 2.9714 px`, improved/regressed `8 / 0` |

Each v9 run uses `22` strict candidates per sample.

## What Changed In v9

- Added `support-v9` candidate grid for hard broad scenes.
- Added selector `v6`, still raw/depth-only.
- Added support-gated LWIR candidates:
  - `v9_depth_thermal` for reflection/background and double-edge mismatch;
  - `v9_target_only` for weak target support;
  - `v9_target_silhouette` for two-target/background-contamination cases.
- Reduced ghosting by suppressing unreliable background/reflection thermal edges outside RGB/depth/thermal support.
- Kept aligned images out of generation and selection; aligned images only evaluate the already-generated candidate.

## Honest Boundary

v9 selected LWIR is sometimes a support-gated registration evidence image, not a claim that every background thermal pixel is physically reconstructed. This is intentional for reflection, weak-target, and background-contaminated scenes: unreliable background thermal edges are suppressed so the panel shows the target registration evidence clearly.

The reports keep this explicit with:

- `phase29_support_mode`;
- `phase29_v9_gate_mode`;
- selector debug CSVs;
- oracle panels marked evaluation-only;
- reliability labels and risk maps.

## Key Outputs

| Folder | Purpose |
|---|---|
| `five_panels/` | Main five-panel review: generated RGB, selected LWIR, RGB/LWIR/depth contours, fusion, risk map. |
| `acceptance_summary_panels/` | One-page acceptance evidence per sample. |
| `selected_clean_lwir/` | Clean selected LWIR or support-gated LWIR evidence. |
| `selected_fusion_review/` | Anti-ghost fusion review. |
| `contour_overlays/` | RGB/LWIR/depth contour alignment overlays. |
| `tear_ghost_maps/` | Tear, ghost, and depth-risk maps. |
| `oracle_ceiling_panels/` | Evaluation-only selected-vs-oracle candidate comparison. |
| `selector_debug/` | Per-candidate raw/depth selection decisions. |
| `metrics/` | `p29_metrics.csv`, `p29_candidates.csv`, `p29_summary.csv`, `p29_v5_reliability.csv`, `p29_best.json`. |
| `reports/` | Markdown acceptance/research report. |

## Reproduce Current v9

```powershell
python .\darklight_mm5\calibration_only_method\phase29\run_phase29.py --run-profile core --candidate-grid support-v9 --version-label v9 --selector-version v6 --output .\darklight_mm5\calibration_only_method\phase29\outputs_core_generalization_v9 --report-level research --save-selector-debug
python .\darklight_mm5\calibration_only_method\phase29\run_phase29.py --run-profile review --candidate-grid support-v9 --version-label v9 --selector-version v6 --output .\darklight_mm5\calibration_only_method\phase29\outputs_review_generalization_v9 --report-level research --save-selector-debug
python .\darklight_mm5\calibration_only_method\phase29\run_phase29.py --run-profile broad --candidate-grid support-v9 --version-label v9 --selector-version v6 --output .\darklight_mm5\calibration_only_method\phase29\outputs_broad_generalization_v9 --report-level research --save-selector-debug
```

Validation:

```powershell
python -m py_compile .\darklight_mm5\calibration_only_method\phase29\run_phase29.py
python -m json.tool .\darklight_mm5\calibration_only_method\phase29\outputs_core_generalization_v9\metrics\p29_best.json
python -m json.tool .\darklight_mm5\calibration_only_method\phase29\outputs_review_generalization_v9\metrics\p29_best.json
python -m json.tool .\darklight_mm5\calibration_only_method\phase29\outputs_broad_generalization_v9\metrics\p29_best.json
```

## Previous Baselines

| Version | broad result | Note |
|---|---:|---|
| Phase28 | `11/18`, edge mean/max `2.8978 / 5.4010 px` | Stable visual baseline. |
| Phase29 v3 | `13/18`, edge mean/max `2.7976 / 4.7836 px` | First honest broad-generalization version. |
| Phase29 v4/v5 | `15/18`, edge mean/max about `2.7202 / 4.7836 px` | Added selector/reliability evidence. |
| Phase29 v6 acceptance-lite | `15/18`, edge mean/max `2.6283 / 4.7164 px` | Small `13`-candidate stable package; kept as documented memory after v9 promotion. |
| Phase29 v7/v8 probes | research only | Larger or component pools exposed ceiling/selection behavior but were not promoted. |

Superseded result bodies are deleted from the active workspace after their lessons are captured here. Current generated acceptance outputs kept in this folder are limited to `outputs_core_generalization_v9/`, `outputs_review_generalization_v9/`, and `outputs_broad_generalization_v9/`.

## Cleanup Memory

The following generated folders were intentionally removed during workspace simplification:

- archived Phase29 v3/probe bodies previously under `_archived_outputs/`;
- v4/v5/v6 generated outputs;
- acceptance-lite generated outputs;
- rejected v7/v8 probe outputs;
- temporary v9 support/selector probes.

The code path remains reproducible through `run_phase29.py`; the historical metrics above preserve why each older output body was superseded.

## Current Acceptance Statement

Phase29 v9 completes the defined broad pressure set under the strict raw/calibration/depth boundary: core `3/3`, review `7/7`, broad `18/18`, and no selected regressions. It should be described as a strict support-gated registration/evidence method, not as a full-scene thermal reconstruction method.
