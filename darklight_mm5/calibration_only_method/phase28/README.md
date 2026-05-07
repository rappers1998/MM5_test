# Phase28 Visual Acceptance Baseline

Phase28 is the stable visual baseline for calibration/depth registration. It shows whether registration is visually credible through depth support, clean LWIR, fusion, contour overlays, and tear/ghost risk maps.

For the latest broad-generalization acceptance package, use `../phase29/README.md`.

## Boundary

Generation uses:

- `calibration/def_stereocalib_THERM.yml`
- `calibration/def_thermalcam_ori.yml`
- raw RGB1
- raw LWIR16
- raw depth
- original calibration-board captures and calibration-derived helpers

MM5 aligned RGB/T16 are evaluation-only. They are not used to choose generation parameters.

## Results

| Profile | Samples | Edge mean/max | Status |
|---|---|---:|---|
| core | `106,104,103` | `2.1019 / 2.7135 px` | pass |
| review | `2,23,103,106,273,291,302` | `2.0386 / 2.7155 px` | pass |
| broad | pressure set | `2.8978 / 5.4010 px` | diagnostic, `11/18` pass |

Phase28 is useful as historical baseline code, but its generated output folders are not kept in the simplified active workspace. Phase29 v9 is the current acceptance route.

## Historical Output Layout

These folders are produced if Phase28 is rerun, but they are treated as disposable generated artifacts after Phase29 v9 promotion.

| Folder | Purpose |
|---|---|
| `outputs_visual_acceptance/` | Core visual acceptance package. |
| `outputs_visual_all/` | Core/review/broad visual package. |
| `acceptance_panels/` | Main visual panels. |
| `clean_registered_lwir/` | Target-focused registered LWIR. |
| `fusion_review/` | Anti-ghost fusion review image. |
| `contour_overlays/` | RGB/LWIR/depth contour overlays. |
| `tear_ghost_maps/` | Tear and ghost risk maps. |
| `edge_error_heatmaps/` | Evaluation-only edge error heatmaps. |
| `metrics/`, `reports/` | CSV/JSON metrics and Markdown reports. |

## Reproduce

```powershell
python .\darklight_mm5\calibration_only_method\phase28\run_phase28.py --run-profile core --output .\darklight_mm5\calibration_only_method\phase28\outputs_visual_acceptance
python .\darklight_mm5\calibration_only_method\phase28\run_phase28.py --run-profile all --output .\darklight_mm5\calibration_only_method\phase28\outputs_visual_all
```

## Removed Experiment Ideas

| Old route | Core idea | Why not current |
|---|---|---|
| Phase25 edge-opt diagnostic | Sweep residual LWIR translation against evaluation-only aligned edge distance. | Aligned metric influenced selection, so it is diagnostic only. |
| Phase26 edge-distance promoted | Promote the Phase25 diagnostic shift as fixed geometry. | Visually weaker and not strong enough for acceptance. |
| Phase27 oracle under-3 study | Use aligned/oracle candidates to prove metric floor. | Violates strict generation boundary. |
| Phase27 calibrated baseline | Calibration-only geometry with fixed `dx=3.5`, `dy=2.0`, `9 px` LWIR stabilization. | Core/review were good, but visual proof and broad diagnostics were weaker than Phase28/29. |
| Phase34 target-locked under-3 | Directly use aligned RGB/T16 as output. | Only an upper bound; invalid as a strict method. |

The old experiment bodies are not active. Their ideas are preserved here and in the planning logs.
