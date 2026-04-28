# Calibration Depth Tuning Design

## Goal
Improve the `calibration_depth` strategy so its registration and fusion quality approaches the current `official_reference` output while keeping deployment free of official aligned image dependencies.

## Approved Approach
Use `official_reference` as an offline tuning target only. The final runtime path still uses:

- `calibration/def_stereocalib_THERM.yml`
- raw RGB1
- raw LWIR
- raw depth

## Optimization Scope
Keep changes focused on:

- scaling/cropping calibration intrinsics to the actual raw RGB/LWIR image sizes;
- small depth-scale and extrinsic translation adjustments;
- bounded residual translation refinement;
- metrics and reports that compare tuned candidates against the current baseline.

Do not revive the old method1-method5 pipelines.

## Success Criteria

- The tuning command runs on the three selected dark-light samples.
- It writes a CSV/JSON summary of candidate parameters and selected best parameters.
- The selected tuned run improves or preserves the current `calibration_depth` score on the three samples.
- The final tuned output directory contains quads, five-panels, per-sample files, and metrics.
- `official_reference` is not required when running the final tuned `calibration_depth` path.

## Implementation Plan

1. Add calibration adjustment helpers to `run_darklight.py`.
2. Add CLI flags for LWIR calibration size, optional tuned intrinsics/extrinsics, and residual shift limit.
3. Add a small offline tuning script that searches a conservative parameter grid and scores against raw RGB/LWIR alignment metrics.
4. Run the search on aligned IDs `106,104,103`.
5. Re-run `calibration_depth` with the best global parameters into `darklight_mm5/outputs_calibration_tuned`.
6. Update README and planning logs with the chosen parameters and results.
