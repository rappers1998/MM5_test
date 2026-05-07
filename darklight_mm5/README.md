# Dark-Light MM5 Workspace

This workspace contains MM5 RGB/LWIR/depth registration, fusion, and evaluation work.

## Current Mainline

| Route | Entry | Purpose |
|---|---|---|
| Phase29 v9 support-gated | `calibration_only_method/phase29/` | Current strict raw/calibration/depth broad-generalization acceptance package. |
| Phase28 | `calibration_only_method/phase28/` | Historical stable visual baseline and helper code for Phase29. |
| Phase25 helpers | `calibration_only_method/run_phase25_depth_assisted.py` and `run_phase25_edge_optimization.py` | Geometry/depth helper provenance reused by later phases. |

Phase29 generation and selection use calibration files, raw RGB, raw LWIR, raw depth, calibration-board geometry, and raw/depth support masks only. MM5 aligned RGB/T16 are evaluation-only.

## Phase29 v9 Snapshot

```text
calibration_only_method/phase29/outputs_core_generalization_v9/
calibration_only_method/phase29/outputs_review_generalization_v9/
calibration_only_method/phase29/outputs_broad_generalization_v9/
```

| Profile | Result |
|---|---:|
| core | `3/3` pass, edge mean/max `1.7249 / 1.8926 px` |
| review | `7/7` pass, edge mean/max `1.6379 / 2.7155 px` |
| broad | `18/18` pass, edge mean/max `2.2408 / 2.9714 px`, improved/regressed `8 / 0`, `22` candidates/sample |

The previous broad failures `050`, `110`, and `123` now pass under the strict selected path. This remains an honest support-gated registration/evidence method, not a full-scene thermal reconstruction claim.

## Review First

| Path | Purpose |
|---|---|
| `calibration_only_method/phase29/outputs_broad_generalization_v9/five_panels/` | Five-panel visual acceptance images. |
| `calibration_only_method/phase29/outputs_broad_generalization_v9/acceptance_summary_panels/` | Main visual acceptance panels. |
| `calibration_only_method/phase29/outputs_broad_generalization_v9/reliability_maps/` | Reliability/risk overlays. |
| `calibration_only_method/phase29/outputs_broad_generalization_v9/hard_ceiling_panels/` | Selected-vs-best-strict evidence. |
| `calibration_only_method/phase29/outputs_broad_generalization_v9/reports/` | Markdown acceptance reports. |
| `calibration_only_method/phase29/README.md` | Superseded Phase29 method memory and reproduce commands. |

## Reproduce

```powershell
python .\darklight_mm5\calibration_only_method\phase29\run_phase29.py --run-profile core --version-label v9 --selector-version v6 --candidate-grid support-v9 --output .\darklight_mm5\calibration_only_method\phase29\outputs_core_generalization_v9 --report-level research --save-selector-debug
python .\darklight_mm5\calibration_only_method\phase29\run_phase29.py --run-profile review --version-label v9 --selector-version v6 --candidate-grid support-v9 --output .\darklight_mm5\calibration_only_method\phase29\outputs_review_generalization_v9 --report-level research --save-selector-debug
python .\darklight_mm5\calibration_only_method\phase29\run_phase29.py --run-profile broad --version-label v9 --selector-version v6 --candidate-grid support-v9 --output .\darklight_mm5\calibration_only_method\phase29\outputs_broad_generalization_v9 --report-level research --save-selector-debug --save-edge-debug
```

## Folder Policy

The active workspace keeps only the Phase29 v9 generated output bodies needed for acceptance: core, review, and broad. Superseded generated folders from Phase25, Phase28, Phase29 v3-v8, temporary v9 probes, teacher residuals, calibration-plane experiments, benchmark method outputs, and local HLS builds are disposable after their conclusions are recorded in README/planning logs.

## Version Memory

| Version | Memory |
|---|---|
| Phase28 | Stable visual baseline; broad `11/18`, edge mean/max `2.8978 / 5.4010 px`. |
| Phase29 v3 | Honest first broad version; broad `13/18`. |
| Phase29 v4/v5 | Broad `15/18`; v5 added reliability maps, hard-ceiling labels, and selector explanations. |
| Phase29 v6 acceptance-lite | Compact pre-v9 selector; broad `15/18`, edge mean/max `2.6283 / 4.7164 px`. |
| Phase29 v7/v8 | Rejected research probes; large/component pools were unstable or insufficient. |
| Phase29 v9 | Current accepted support-gated method; broad `18/18`, edge mean/max `2.2408 / 2.9714 px`. |
