# Calibration-Only MM5 Registration

This directory contains the strict calibration-only MM5 registration path. The current accepted package is Phase29 v9 with the `support-v9` strict support-gated grid. Phase28, Phase25, and earlier Phase29 versions remain as source/provenance history, while their generated output bodies are no longer part of the active acceptance workspace.

## Current Path

```text
phase29/run_phase29.py
```

Recommended evidence:

- `phase29/README.md`
- `phase29/outputs_broad_generalization_v9/reports/p29_broad_generalization_report.md`
- `phase29/outputs_broad_generalization_v9/five_panels/`
- `phase29/outputs_broad_generalization_v9/acceptance_summary_panels/`
- `phase29/outputs_broad_generalization_v9/oracle_ceiling_panels/`
- `phase29/outputs_broad_generalization_v9/reliability_maps/`
- `phase29/outputs_broad_generalization_v9/selector_debug/`
- `phase29/README.md` for superseded method history
- `phase28/README.md`

## Data Boundary

Generation and selection may use:

- MM5 calibration files under `../../calibration/`;
- raw RGB1, raw LWIR16, and raw depth from the MM5 index;
- original calibration-board captures;
- calibration-derived board correspondences, depth support, anti-ghost masks, and risk maps.

Generation and raw-only selection must not use:

- MM5 aligned RGB/T16 as a parameter source;
- teacher residuals, official aligned transforms, or aligned templates;
- per-sample fitting that reads aligned images.

Aligned images are evaluation-only for NCC, edge distance, heatmaps, candidate ceiling, and acceptance comparisons.

## Results

| Version | core | review | broad |
|---|---:|---:|---:|
| Phase28 baseline | `3/3` | `7/7` | `11/18`, edge mean/max `2.8978 / 5.4010 px` |
| Phase29 v3 | `3/3` | `7/7` | `13/18`, edge mean/max `2.7976 / 4.7836 px` |
| Phase29 v4 | `3/3` | `7/7` | `15/18`, edge mean/max `2.7202 / 4.7836 px` |
| Phase29 v5 | `3/3` | `7/7` | `15/18`, edge mean/max `2.7202 / 4.7836 px` |
| Phase29 v6 acceptance-lite | `3/3` | `7/7` | `15/18`, edge mean/max `2.6283 / 4.7164 px`, `13` candidates/sample |
| Phase29 v9 support-gated | `3/3`, edge mean/max `1.7249 / 1.8926 px` | `7/7`, edge mean/max `1.6379 / 2.7155 px` | `18/18`, edge mean/max `2.2408 / 2.9714 px`, `22` candidates/sample |

Phase29 v9 selected improved/regressed count on broad is `8 / 0`. It solves the previous broad failures `050`, `110`, and `123`, and also keeps `187` under `3 px`, by using raw/depth support-gated LWIR evidence rather than pretending unreliable background thermal edges are valid.

## Reproduce

```powershell
python .\darklight_mm5\calibration_only_method\phase29\run_phase29.py --run-profile core --candidate-grid support-v9 --version-label v9 --selector-version v6 --output .\darklight_mm5\calibration_only_method\phase29\outputs_core_generalization_v9 --report-level research --save-selector-debug
python .\darklight_mm5\calibration_only_method\phase29\run_phase29.py --run-profile review --candidate-grid support-v9 --version-label v9 --selector-version v6 --output .\darklight_mm5\calibration_only_method\phase29\outputs_review_generalization_v9 --report-level research --save-selector-debug
python .\darklight_mm5\calibration_only_method\phase29\run_phase29.py --run-profile broad --candidate-grid support-v9 --version-label v9 --selector-version v6 --output .\darklight_mm5\calibration_only_method\phase29\outputs_broad_generalization_v9 --report-level research --save-selector-debug
```

## Kept Structure

| Path | Status |
|---|---|
| `phase29/` | Current strict broad-generalization and support-gated acceptance version. |
| `phase28/` | Historical visual baseline and helper code imported by Phase29. |
| `run_phase25_edge_optimization.py` | Helper reused by Phase28/29. |
| `run_phase25_depth_assisted.py` | Phase25 geometry provenance helper. |
| `run_phase21_canvas_optimization.py` to `run_phase24_lwir_board_affine.py` | Calibration and board-affine helpers. |
| `run_calibration_only.py`, `diagnose_aligned_canvas.py` | Historical diagnostic entrypoints. |

Generated outputs kept for active acceptance are limited to Phase29 v9 core/review/broad. Superseded output bodies from Phase25, Phase28, Phase29 v3-v8, temporary v9 probes, and older diagnostics are deleted after their metrics and lessons are recorded in README/planning logs.

## Historical Notes

| Version | Outcome |
|---|---|
| Phase25 | Calibration-only depth-assisted foundation; useful for geometry and helper functions, not final acceptance. |
| Phase28 | Stable visual baseline; broad `11/18`, edge mean/max `2.8978 / 5.4010 px`. |
| Phase29 v3 | First honest broad-generalization selector; broad `13/18`. |
| Phase29 v4/v5 | Broad `15/18`; v5 added explainability/reliability evidence. |
| Phase29 v6 acceptance-lite | Best compact pre-v9 selector; broad `15/18`, edge mean/max `2.6283 / 4.7164 px`. |
| Phase29 v7/v8 | Rejected research probes; larger/component candidate pools did not promote safely. |
| Phase29 v9 | Current support-gated acceptance package; broad `18/18`, edge mean/max `2.2408 / 2.9714 px`. |
