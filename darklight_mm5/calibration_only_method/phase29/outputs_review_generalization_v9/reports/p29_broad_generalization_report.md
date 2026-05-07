# Phase29 Broad-Generalization Report

## Boundary
- Generation uses calibration files, raw RGB/LWIR, raw depth, and calibration-derived helpers.
- MM5 aligned RGB/T16 are evaluation-only.
- Candidate selection uses raw-only reliability scores and does not read aligned metrics.

## Profile
- profile: `review`
- aligned ids: `2,23,103,106,273,291,302`
- selector version: `v6`
- candidate grid: `support-v9`
- run mode: `strict-selected`
- report level: `research`
- version label: `v9`
- explainability level: `full`
- reliability gate: `strict`
- failure focus: `all`

## Result
- selected edge mean/max: `1.6379` / `2.7155` px
- selected LWIR NCC mean/min: `0.9363` / `0.9188`
- pass/fail: `7` / `0`
- improved/regressed vs Phase28 baseline: `3` / `0`
- raw score mean/min/max: `19.7002` / `12.7881` / `41.3399`
- reliability score mean/min: `79.17` / `64.04`
- reliability labels: `accepted=3`, `improved=0`, `risky-pass=4`, `hard-ceiling-fail=0`, `selector-gap-fail=0`

## Evaluation-Only Candidate Ceiling
- This ceiling is computed after every strict candidate is generated; it is not used for generation or raw-only selection.
- ceiling edge mean/max: `1.2595` / `1.6368` px
- ceiling pass/fail: `7` / `0`
- candidate-pool-unsolved failures: `0`
- raw-selector gap failures: `0`

## Per-Sample Selection
- `002_seq283`: selected `p29_baseline`, reliability `accepted` `84.1`, scene `general`, raw `14.567`, edge `1.2045px`, baseline `1.2045px`, delta `+0.0000px`, oracle `p29_raw_shift_dx2p5_dy2p0_fill` `1.1106px`, reason `selector v6 retained baseline; best raw improvement 0.7997; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: raw improvement 0.2051 is below 0.6500; +18 more rejected)`
- `023_seq305`: selected `p29_baseline`, reliability `accepted` `86.2`, scene `general`, raw `13.868`, edge `1.4078px`, baseline `1.4078px`, delta `+0.0000px`, oracle `p29_raw_shift_dx2p5_dy2p0_fill` `1.3622px`, reason `selector v6 retained baseline; best raw improvement 0.1647; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: raw improvement -0.1222 is below 0.6500; +18 more rejected)`
- `103_seq385`: selected `p29_v9_target_only`, reliability `risky-pass` `78.0`, scene `general`, raw `41.340`, edge `1.8926px`, baseline `2.7135px`, delta `+0.8209px`, oracle `p29_v9_support_gated_tight` `1.6368px`, reason `selector v6 selected v9_support_gated via v9_weak_target_only; raw_score delta +23.7090; adjusted score 1.1953; selection used raw RGB/LWIR/depth support gates only`
- `106_seq388`: selected `p29_baseline`, reliability `accepted` `85.8`, scene `general`, raw `12.788`, edge `1.4009px`, baseline `1.4009px`, delta `+0.0000px`, oracle `p29_raw_shift_dx2p5_dy2p0_fill` `1.2276px`, reason `selector v6 retained baseline; best raw improvement 0.2206; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: raw improvement -0.0417 is below 0.6500; +18 more rejected)`
- `273_seq559`: selected `p29_v9_target_only`, reliability `risky-pass` `78.0`, scene `general`, raw `15.504`, edge `1.3594px`, baseline `2.2384px`, delta `+0.8790px`, oracle `p29_v9_support_gated_tight` `1.1014px`, reason `selector v6 selected v9_support_gated via v9_weak_target_only; raw_score delta -0.6146; adjusted score 1.7722; selection used raw RGB/LWIR/depth support gates only`
- `291_seq577`: selected `p29_baseline`, reliability `risky-pass` `64.0`, scene `double_edge_mismatch`, raw `19.391`, edge `2.7155px`, baseline `2.7155px`, delta `+0.0000px`, oracle `p29_v9_support_gated` `1.2593px`, reason `selector v6 retained baseline; best raw improvement 0.9551; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: LWIR edge growth 1.127 exceeds 1.080; +18 more rejected)`
- `302_seq588`: selected `p29_v9_target_only`, reliability `risky-pass` `78.0`, scene `general`, raw `20.444`, edge `1.4846px`, baseline `2.5896px`, delta `+1.1049px`, oracle `p29_v9_support_gated` `1.1187px`, reason `selector v6 selected v9_support_gated via v9_weak_target_only; raw_score delta +6.8031; adjusted score 1.2427; selection used raw RGB/LWIR/depth support gates only`

## Selector Debug Summary
- `002_seq283`: rule `v6_baseline`, eligible `True`, adjusted `14.5669`, raw_improvement `0.0000`, edge_growth `1.000`, shift `(+0.00,+0.00)`
- `023_seq305`: rule `v6_baseline`, eligible `True`, adjusted `13.8679`, raw_improvement `0.0000`, edge_growth `1.000`, shift `(+0.00,+0.00)`
- `103_seq385`: rule `v9_weak_target_only`, eligible `True`, adjusted `1.1953`, raw_improvement `-23.7090`, edge_growth `0.443`, shift `(+0.00,+0.00)`
- `106_seq388`: rule `v6_baseline`, eligible `True`, adjusted `12.7881`, raw_improvement `0.0000`, edge_growth `1.000`, shift `(+0.00,+0.00)`
- `273_seq559`: rule `v9_weak_target_only`, eligible `True`, adjusted `1.7722`, raw_improvement `0.6146`, edge_growth `0.992`, shift `(+0.00,+0.00)`
- `291_seq577`: rule `v6_baseline`, eligible `True`, adjusted `19.3910`, raw_improvement `0.0000`, edge_growth `1.000`, shift `(+0.00,+0.00)`
- `302_seq588`: rule `v9_weak_target_only`, eligible `True`, adjusted `1.2427`, raw_improvement `-6.8031`, edge_growth `0.457`, shift `(+0.00,+0.00)`

## Remaining Failures
- none

## Hard-Ceiling Interpretation
- No selected failure is blocked by the current strict candidate pool.
- No remaining failure has a hidden passing strict candidate in the current pool.

## Candidate Evidence
- total candidate rows: `154`
- `metrics/p29_candidates.csv` contains raw-only scores and aligned-only post-selection evaluation columns.
- `candidate_panels/` shows the top raw-scored candidates per sample for audit.
- `oracle_ceiling_panels/` compares the selected result with the evaluation-only best strict candidate.
- `acceptance_summary_panels/` is the v5 one-page visual acceptance evidence for every sample.
- `hard_ceiling_panels/` shows selected-vs-ceiling evidence for hard-limit review.
- `reliability_maps/` overlays selected support and risk using the v5 reliability label.
- `failure_explanations/` contains per-sample Markdown explanations.
- `selector_debug/` is written when `--save-selector-debug` is enabled.
- `before_after_phase28_phase29/` compares the Phase28 baseline and Phase29 selected result.
