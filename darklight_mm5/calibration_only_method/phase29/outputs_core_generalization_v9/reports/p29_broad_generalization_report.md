# Phase29 Broad-Generalization Report

## Boundary
- Generation uses calibration files, raw RGB/LWIR, raw depth, and calibration-derived helpers.
- MM5 aligned RGB/T16 are evaluation-only.
- Candidate selection uses raw-only reliability scores and does not read aligned metrics.

## Profile
- profile: `core`
- aligned ids: `106,104,103`
- selector version: `v6`
- candidate grid: `support-v9`
- run mode: `strict-selected`
- report level: `research`
- version label: `v9`
- explainability level: `full`
- reliability gate: `strict`
- failure focus: `all`

## Result
- selected edge mean/max: `1.7249` / `1.8926` px
- selected LWIR NCC mean/min: `0.9212` / `0.8954`
- pass/fail: `3` / `0`
- improved/regressed vs Phase28 baseline: `2` / `0`
- raw score mean/min/max: `27.6625` / `12.7881` / `41.3399`
- reliability score mean/min: `80.62` / `78.00`
- reliability labels: `accepted=1`, `improved=0`, `risky-pass=2`, `hard-ceiling-fail=0`, `selector-gap-fail=0`

## Evaluation-Only Candidate Ceiling
- This ceiling is computed after every strict candidate is generated; it is not used for generation or raw-only selection.
- ceiling edge mean/max: `1.4377` / `1.6368` px
- ceiling pass/fail: `3` / `0`
- candidate-pool-unsolved failures: `0`
- raw-selector gap failures: `0`

## Per-Sample Selection
- `103_seq385`: selected `p29_v9_target_only`, reliability `risky-pass` `78.0`, scene `general`, raw `41.340`, edge `1.8926px`, baseline `2.7135px`, delta `+0.8209px`, oracle `p29_v9_support_gated_tight` `1.6368px`, reason `selector v6 selected v9_support_gated via v9_weak_target_only; raw_score delta +23.7090; adjusted score 1.1953; selection used raw RGB/LWIR/depth support gates only`
- `104_seq386`: selected `p29_v9_target_only`, reliability `risky-pass` `78.0`, scene `general`, raw `28.860`, edge `1.8814px`, baseline `2.1912px`, delta `+0.3098px`, oracle `p29_raw_shift_dx2p5_dy3p0_fill` `1.4487px`, reason `selector v6 selected v9_support_gated via v9_weak_target_only; raw_score delta +15.1334; adjusted score 1.1011; selection used raw RGB/LWIR/depth support gates only`
- `106_seq388`: selected `p29_baseline`, reliability `accepted` `85.8`, scene `general`, raw `12.788`, edge `1.4009px`, baseline `1.4009px`, delta `+0.0000px`, oracle `p29_raw_shift_dx2p5_dy2p0_fill` `1.2276px`, reason `selector v6 retained baseline; best raw improvement 0.2206; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: raw improvement -0.0417 is below 0.6500; +18 more rejected)`

## Selector Debug Summary
- `103_seq385`: rule `v9_weak_target_only`, eligible `True`, adjusted `1.1953`, raw_improvement `-23.7090`, edge_growth `0.443`, shift `(+0.00,+0.00)`
- `104_seq386`: rule `v9_weak_target_only`, eligible `True`, adjusted `1.1011`, raw_improvement `-15.1334`, edge_growth `0.324`, shift `(+0.00,+0.00)`
- `106_seq388`: rule `v6_baseline`, eligible `True`, adjusted `12.7881`, raw_improvement `0.0000`, edge_growth `1.000`, shift `(+0.00,+0.00)`

## Remaining Failures
- none

## Hard-Ceiling Interpretation
- No selected failure is blocked by the current strict candidate pool.
- No remaining failure has a hidden passing strict candidate in the current pool.

## Candidate Evidence
- total candidate rows: `66`
- `metrics/p29_candidates.csv` contains raw-only scores and aligned-only post-selection evaluation columns.
- `candidate_panels/` shows the top raw-scored candidates per sample for audit.
- `oracle_ceiling_panels/` compares the selected result with the evaluation-only best strict candidate.
- `acceptance_summary_panels/` is the v5 one-page visual acceptance evidence for every sample.
- `hard_ceiling_panels/` shows selected-vs-ceiling evidence for hard-limit review.
- `reliability_maps/` overlays selected support and risk using the v5 reliability label.
- `failure_explanations/` contains per-sample Markdown explanations.
- `selector_debug/` is written when `--save-selector-debug` is enabled.
- `before_after_phase28_phase29/` compares the Phase28 baseline and Phase29 selected result.
