# Phase29 Broad-Generalization Report

## Boundary
- Generation uses calibration files, raw RGB/LWIR, raw depth, and calibration-derived helpers.
- MM5 aligned RGB/T16 are evaluation-only.
- Candidate selection uses raw-only reliability scores and does not read aligned metrics.

## Profile
- profile: `broad`
- aligned ids: `200,209,248,291,161,187,137,103,23,50,100,272,266,262,296,123,110,120`
- selector version: `v6`
- candidate grid: `support-v9`
- run mode: `strict-selected`
- report level: `research`
- version label: `v9`
- explainability level: `full`
- reliability gate: `strict`
- failure focus: `all`

## Result
- selected edge mean/max: `2.2408` / `2.9714` px
- selected LWIR NCC mean/min: `0.9259` / `0.8747`
- pass/fail: `18` / `0`
- improved/regressed vs Phase28 baseline: `8` / `0`
- raw score mean/min/max: `18.1479` / `12.6145` / `41.3399`
- reliability score mean/min: `72.13` / `62.55`
- reliability labels: `accepted=5`, `improved=1`, `risky-pass=12`, `hard-ceiling-fail=0`, `selector-gap-fail=0`

## Evaluation-Only Candidate Ceiling
- This ceiling is computed after every strict candidate is generated; it is not used for generation or raw-only selection.
- ceiling edge mean/max: `1.6079` / `2.7905` px
- ceiling pass/fail: `18` / `0`
- candidate-pool-unsolved failures: `0`
- raw-selector gap failures: `0`

## Per-Sample Selection
- `023_seq305`: selected `p29_baseline`, reliability `accepted` `86.2`, scene `general`, raw `13.868`, edge `1.4078px`, baseline `1.4078px`, delta `+0.0000px`, oracle `p29_raw_shift_dx2p5_dy2p0_fill` `1.3622px`, reason `selector v6 retained baseline; best raw improvement 0.1647; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: raw improvement -0.1222 is below 0.6500; +18 more rejected)`
- `050_seq332`: selected `p29_v9_depth_thermal`, reliability `risky-pass` `78.0`, scene `reflection_background`, raw `18.774`, edge `1.5848px`, baseline `5.4010px`, delta `+3.8162px`, oracle `p29_v9_support_gated` `1.5722px`, reason `selector v6 selected v9_support_gated via v9_reflection_depth_thermal; raw_score delta -1.4893; adjusted score 0.3465; selection used raw RGB/LWIR/depth support gates only`
- `100_seq382`: selected `p29_baseline`, reliability `accepted` `83.5`, scene `reflection_background`, raw `12.614`, edge `1.7430px`, baseline `1.7430px`, delta `+0.0000px`, oracle `p29_small_target_guard` `1.6708px`, reason `selector v6 retained baseline; best raw improvement 0.3057; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: raw improvement -0.0891 is below 0.6500; +18 more rejected)`
- `103_seq385`: selected `p29_v9_target_only`, reliability `risky-pass` `78.0`, scene `general`, raw `41.340`, edge `1.8926px`, baseline `2.7135px`, delta `+0.8209px`, oracle `p29_v9_support_gated_tight` `1.6368px`, reason `selector v6 selected v9_support_gated via v9_weak_target_only; raw_score delta +23.7090; adjusted score 1.1953; selection used raw RGB/LWIR/depth support gates only`
- `110_seq396`: selected `p29_v9_target_only`, reliability `improved` `74.0`, scene `general`, raw `18.213`, edge `2.2281px`, baseline `3.5698px`, delta `+1.3417px`, oracle `p29_v9_target_only` `2.2281px`, reason `selector v6 selected v9_support_gated via v9_weak_target_only; raw_score delta +2.1748; adjusted score 1.6327; selection used raw RGB/LWIR/depth support gates only`
- `120_seq406`: selected `p29_raw_shift_dx3p5_dy1p5_fill`, reliability `risky-pass` `62.5`, scene `depth_tearing`, raw `17.001`, edge `2.6687px`, baseline `3.1005px`, delta `+0.4318px`, oracle `p29_raw_shift_dx3p5_dy1p5_fill` `2.6687px`, reason `selector v6 selected raw_shift_fill via depth_tearing_v2; raw_score delta -0.1308; adjusted score 24.4086; selection used raw RGB/LWIR/depth support gates only`
- `123_seq409`: selected `p29_v9_target_silhouette`, reliability `risky-pass` `78.0`, scene `general`, raw `21.836`, edge `1.2253px`, baseline `4.7836px`, delta `+3.5583px`, oracle `p29_v9_target_silhouette` `1.2253px`, reason `selector v6 selected v9_support_gated via v9_target_silhouette; raw_score delta +7.8210; adjusted score 2.3043; selection used raw RGB/LWIR/depth support gates only`
- `137_seq423`: selected `p29_baseline`, reliability `accepted` `70.0`, scene `general`, raw `13.903`, edge `2.6714px`, baseline `2.6714px`, delta `+0.0000px`, oracle `p29_raw_shift_dx4p0_dy2p0_fill` `2.5929px`, reason `selector v6 retained baseline; best raw improvement 3.4869; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: raw improvement -0.4174 is below 0.6500; +18 more rejected)`
- `161_seq447`: selected `p29_baseline`, reliability `risky-pass` `63.4`, scene `weak_target_support`, raw `18.087`, edge `2.9714px`, baseline `2.9714px`, delta `+0.0000px`, oracle `p29_v9_target_only` `1.3156px`, reason `selector v6 retained baseline; best raw improvement 3.2657; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: LWIR edge growth 1.122 exceeds 1.080; +18 more rejected)`
- `187_seq473`: selected `p29_v9_depth_thermal`, reliability `risky-pass` `65.2`, scene `double_edge_mismatch`, raw `13.603`, edge `2.7905px`, baseline `3.6135px`, delta `+0.8230px`, oracle `p29_v9_depth_thermal` `2.7905px`, reason `selector v6 selected v9_support_gated via v9_depth_thermal_double_edge; raw_score delta -9.6659; adjusted score 4.2234; selection used raw RGB/LWIR/depth support gates only`
- `200_seq486`: selected `p29_baseline`, reliability `risky-pass` `68.2`, scene `double_edge_mismatch`, raw `16.749`, edge `2.7090px`, baseline `2.7090px`, delta `+0.0000px`, oracle `p29_v9_support_gated_tight` `1.3574px`, reason `selector v6 retained baseline; best raw improvement 0.1992; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: raw improvement 0.1992 is below 0.6500; +18 more rejected)`
- `209_seq495`: selected `p29_raw_shift_dx4p5_dy3p0_fill`, reliability `risky-pass` `64.5`, scene `general`, raw `16.960`, edge `2.8534px`, baseline `3.4410px`, delta `+0.5876px`, oracle `p29_v9_depth_thermal` `1.4001px`, reason `selector v6 selected raw_shift_fill via generic_v2; raw_score delta -0.6777; adjusted score 17.0605; selection used raw RGB/LWIR/depth support gates only`
- `248_seq534`: selected `p29_baseline`, reliability `risky-pass` `63.1`, scene `reflection_background`, raw `19.324`, edge `2.7664px`, baseline `2.7664px`, delta `+0.0000px`, oracle `p29_v9_support_gated` `1.4168px`, reason `selector v6 retained baseline; best raw improvement 1.9245; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: raw symmetric edge improvement -2.8093 is below 0.0350; +18 more rejected)`
- `262_seq548`: selected `p29_baseline`, reliability `accepted` `83.6`, scene `depth_tearing`, raw `17.656`, edge `1.3865px`, baseline `1.3865px`, delta `+0.0000px`, oracle `p29_v9_target_silhouette` `0.7544px`, reason `selector v6 retained baseline; best raw improvement 0.2081; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: small-target guard is blocked in depth-tearing scenes so local depth correction can be tested; +18 more rejected)`
- `266_seq552`: selected `p29_baseline`, reliability `accepted` `81.7`, scene `depth_tearing`, raw `16.097`, edge `1.7500px`, baseline `1.7500px`, delta `+0.0000px`, oracle `p29_raw_shift_dx3p5_dy1p5_fill` `1.6581px`, reason `selector v6 retained baseline; best raw improvement 1.5931; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: small-target guard is blocked in depth-tearing scenes so local depth correction can be tested; +18 more rejected)`
- `272_seq558`: selected `p29_baseline`, reliability `risky-pass` `70.7`, scene `general`, raw `16.643`, edge `2.2249px`, baseline `2.2249px`, delta `+0.0000px`, oracle `p29_v9_support_gated_tight` `1.1936px`, reason `selector v6 retained baseline; best raw improvement 2.4239; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: LWIR edge growth 1.228 exceeds 1.080; +18 more rejected)`
- `291_seq577`: selected `p29_baseline`, reliability `risky-pass` `64.0`, scene `double_edge_mismatch`, raw `19.391`, edge `2.7155px`, baseline `2.7155px`, delta `+0.0000px`, oracle `p29_v9_support_gated` `1.2593px`, reason `selector v6 retained baseline; best raw improvement 0.9551; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: LWIR edge growth 1.127 exceeds 1.080; +18 more rejected)`
- `296_seq582`: selected `p29_raw_shift_dx4p5_dy1p0_fill`, reliability `risky-pass` `63.7`, scene `weak_target_support`, raw `14.602`, edge `2.7444px`, baseline `3.1907px`, delta `+0.4464px`, oracle `p29_v9_depth_thermal` `0.8395px`, reason `selector v6 selected raw_shift_fill via generic_v2; raw_score delta -0.2443; adjusted score 14.7023; selection used raw RGB/LWIR/depth support gates only`

## Selector Debug Summary
- `023_seq305`: rule `v6_baseline`, eligible `True`, adjusted `13.8679`, raw_improvement `0.0000`, edge_growth `1.000`, shift `(+0.00,+0.00)`
- `050_seq332`: rule `v9_reflection_depth_thermal`, eligible `True`, adjusted `0.3465`, raw_improvement `1.4893`, edge_growth `1.050`, shift `(+0.00,+0.00)`
- `100_seq382`: rule `v6_baseline`, eligible `True`, adjusted `12.6145`, raw_improvement `0.0000`, edge_growth `1.000`, shift `(+0.00,+0.00)`
- `103_seq385`: rule `v9_weak_target_only`, eligible `True`, adjusted `1.1953`, raw_improvement `-23.7090`, edge_growth `0.443`, shift `(+0.00,+0.00)`
- `110_seq396`: rule `v9_weak_target_only`, eligible `True`, adjusted `1.6327`, raw_improvement `-2.1748`, edge_growth `0.798`, shift `(+0.00,+0.00)`
- `120_seq406`: rule `depth_tearing_v2`, eligible `True`, adjusted `24.4086`, raw_improvement `0.1308`, edge_growth `0.923`, shift `(+0.00,-0.50)`
- `123_seq409`: rule `v9_target_silhouette`, eligible `True`, adjusted `2.3043`, raw_improvement `-7.8210`, edge_growth `0.157`, shift `(+0.00,+0.00)`
- `137_seq423`: rule `v6_baseline`, eligible `True`, adjusted `13.9026`, raw_improvement `0.0000`, edge_growth `1.000`, shift `(+0.00,+0.00)`
- `161_seq447`: rule `v6_baseline`, eligible `True`, adjusted `18.0873`, raw_improvement `0.0000`, edge_growth `1.000`, shift `(+0.00,+0.00)`
- `187_seq473`: rule `v9_depth_thermal_double_edge`, eligible `True`, adjusted `4.2234`, raw_improvement `9.6659`, edge_growth `4.387`, shift `(+0.00,+0.00)`
- `200_seq486`: rule `v6_baseline`, eligible `True`, adjusted `16.7489`, raw_improvement `0.0000`, edge_growth `1.000`, shift `(+0.00,+0.00)`
- `209_seq495`: rule `generic_v2`, eligible `True`, adjusted `17.0605`, raw_improvement `0.6777`, edge_growth `0.988`, shift `(+1.00,+1.00)`
- `248_seq534`: rule `v6_baseline`, eligible `True`, adjusted `19.3239`, raw_improvement `0.0000`, edge_growth `1.000`, shift `(+0.00,+0.00)`
- `262_seq548`: rule `v6_baseline`, eligible `True`, adjusted `17.6556`, raw_improvement `0.0000`, edge_growth `1.000`, shift `(+0.00,+0.00)`
- `266_seq552`: rule `v6_baseline`, eligible `True`, adjusted `16.0972`, raw_improvement `0.0000`, edge_growth `1.000`, shift `(+0.00,+0.00)`
- `272_seq558`: rule `v6_baseline`, eligible `True`, adjusted `16.6431`, raw_improvement `0.0000`, edge_growth `1.000`, shift `(+0.00,+0.00)`
- `291_seq577`: rule `v6_baseline`, eligible `True`, adjusted `19.3910`, raw_improvement `0.0000`, edge_growth `1.000`, shift `(+0.00,+0.00)`
- `296_seq582`: rule `generic_v2`, eligible `True`, adjusted `14.7023`, raw_improvement `0.2443`, edge_growth `1.004`, shift `(+1.00,-1.00)`

## Remaining Failures
- none

## Hard-Ceiling Interpretation
- No selected failure is blocked by the current strict candidate pool.
- No remaining failure has a hidden passing strict candidate in the current pool.

## Candidate Evidence
- total candidate rows: `396`
- `metrics/p29_candidates.csv` contains raw-only scores and aligned-only post-selection evaluation columns.
- `candidate_panels/` shows the top raw-scored candidates per sample for audit.
- `oracle_ceiling_panels/` compares the selected result with the evaluation-only best strict candidate.
- `acceptance_summary_panels/` is the v5 one-page visual acceptance evidence for every sample.
- `hard_ceiling_panels/` shows selected-vs-ceiling evidence for hard-limit review.
- `reliability_maps/` overlays selected support and risk using the v5 reliability label.
- `failure_explanations/` contains per-sample Markdown explanations.
- `selector_debug/` is written when `--save-selector-debug` is enabled.
- `before_after_phase28_phase29/` compares the Phase28 baseline and Phase29 selected result.
