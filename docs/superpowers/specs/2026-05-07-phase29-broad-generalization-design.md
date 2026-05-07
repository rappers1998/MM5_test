# Phase29 Broad-Generalization Design

## Goal

Phase29 targets broad pressure-set generalization under the strict generation boundary:

- generation may use calibration files, raw RGB, raw LWIR, raw depth, and calibration-board-derived helpers;
- MM5 aligned RGB/T16 may be read only after generation for evaluation, reports, and heatmaps;
- the method must not use aligned images to choose per-sample parameters;
- if a broad sample cannot be made to pass honestly under this boundary, the report must explain the ceiling instead of hiding the failure.

## Current Phase28 Baseline

Phase28 passes core and review:

- core edge mean/max: `2.1019 / 2.7135 px`;
- review edge mean/max: `2.0386 / 2.7155 px`.

Broad remains 11 pass / 7 fail:

- `050`: reflection/background thermal support and abnormal depth;
- `110`: weak target and ambiguous RGB/LWIR edge evidence;
- `120`: depth discontinuity and tearing risk;
- `123`: full-frame background edge contamination;
- `187`: strong double-edge mismatch;
- `209`: local residual shift mismatch;
- `296`: very small support and unstable edge metric.

## Phase29 Architecture

Phase29 will be a new package:

```text
darklight_mm5/calibration_only_method/phase29/
```

It will reuse Phase28's mature geometry and output conventions, but add a new selection layer:

```text
Phase28 baseline geometry
-> scene diagnostics from raw RGB/LWIR/depth
-> strict allowed candidate generation
-> raw-only reliability scoring
-> selected output
-> aligned-only evaluation
-> honest broad ceiling report
```

## Candidate Families

Phase29 will evaluate several generated candidates per sample before reading aligned evaluation metrics:

1. `p29_baseline`: Phase28 geometry and support policy.
2. `p29_reflection_guard`: stronger rejection of oversized thermal foreground, remote background components, and depth-inconsistent hot regions.
3. `p29_small_target_guard`: fallback support for very small target masks, using compact RGB/thermal consensus instead of broad depth.
4. `p29_depth_conservative`: disables or reduces depth border fill when depth discontinuity or support instability is high.
5. `p29_raw_shift_local`: tests a small fixed grid of local residual shifts and scores candidates using only raw RGB/LWIR/depth consistency.
6. `p29_edge_contamination_guard`: uses target-focused support for selection when full-frame edges are dominated by background.

All candidates are generated from raw and calibration inputs only.

## Raw-Only Selection Score

Phase29 will select the candidate with the lowest raw-only risk score. The score combines:

- RGB/LWIR edge distance inside the generated target support;
- support size stability and compactness;
- thermal foreground area sanity;
- rejected background area;
- depth discontinuity near alpha/fusion boundary;
- ghost-edge ratio from RGB/LWIR edges inside support;
- alpha boundary crossing strong RGB/LWIR/depth edges;
- penalty for extremely small support or oversized support.

Aligned metrics are computed only after selection and are recorded separately.

## Outputs

The main output will be:

```text
phase29/outputs_broad_generalization/
```

Expected subfolders:

- `selected_registered_lwir/`
- `selected_acceptance_panels/`
- `before_after_phase28_phase29/`
- `candidate_panels/`
- `selection_maps/`
- `tear_ghost_maps/`
- `edge_error_heatmaps/`
- `metrics/p29_metrics.csv`
- `metrics/p29_candidates.csv`
- `metrics/p29_summary.csv`
- `metrics/p29_best.json`
- `reports/p29_broad_generalization_report.md`

## Success Criteria

Primary success:

- every Phase28 broad failure must either improve or receive a concrete ceiling explanation;
- broad pass count should improve beyond Phase28's `11/18`;
- core and review must not clearly regress.

Strong success:

- broad reaches all or nearly all samples under `<3 px` without violating generation boundaries;
- reflection and weak-support samples show visually clearer risk maps and fewer misleading fusion artifacts.

Failure handling:

- if any sample remains above `<3 px`, report the raw-only reason and the aligned-only measured gap;
- do not mark a sample as solved solely by masking the evidence away.

## Verification

- `python -m py_compile darklight_mm5/calibration_only_method/phase29/run_phase29.py`
- Run broad profile and validate JSON outputs.
- Run core/review smoke profiles to check regression.
- Check candidate selection CSV contains selected candidate names, raw score components, and post-selection evaluation metrics.
- Check reports mention remaining failures explicitly.
