# Phase29 v5 Explanation: 161_seq447

## Boundary
- Generation and raw selection use calibration files, raw RGB, raw LWIR, and raw depth only.
- MM5 aligned data is used after generation only for evaluation and ceiling reporting.

## Selected Result
- selected candidate: `p29_baseline`
- reliability: `risky-pass` / `63.4`
- selected edge: `2.9714px`
- baseline edge: `2.9714px`
- strict ceiling: `p29_v9_target_only` / `1.3156px`
- scene class: `weak_target_support`
- selector reason: `selector v6 retained baseline; best raw improvement 3.2657; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: LWIR edge growth 1.122 exceeds 1.080; +18 more rejected)`
- reliability reason: `edge close to threshold; weak target support`

## Raw Evidence
- raw score: `18.0873`
- support pixels: `3164`
- rejected background pixels: `568`
- tear risk pixels: `1014`
- ghost edge ratio: `0.8824`
- thermal-depth centroid distance: `27.7823px`

## Ceiling Interpretation
- The selected strict result passes the edge threshold; reliability label describes residual visual risk.

## Visual Files
- acceptance summary: `acceptance_summary_panels/p29_s161_acceptance_summary.png`
- hard ceiling panel: `hard_ceiling_panels/p29_s161_hard_ceiling.png`
- reliability map: `reliability_maps/p29_s161_reliability.png`
- candidate panel: `candidate_panels/p29_s161_candidate_panel.png`
