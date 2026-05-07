# Phase29 v5 Explanation: 272_seq558

## Boundary
- Generation and raw selection use calibration files, raw RGB, raw LWIR, and raw depth only.
- MM5 aligned data is used after generation only for evaluation and ceiling reporting.

## Selected Result
- selected candidate: `p29_baseline`
- reliability: `risky-pass` / `70.7`
- selected edge: `2.2249px`
- baseline edge: `2.2249px`
- strict ceiling: `p29_v9_support_gated_tight` / `1.1936px`
- scene class: `general`
- selector reason: `selector v6 retained baseline; best raw improvement 2.4239; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: LWIR edge growth 1.228 exceeds 1.080; +18 more rejected)`
- reliability reason: `support is fragmented`

## Raw Evidence
- raw score: `16.6431`
- support pixels: `6102`
- rejected background pixels: `1340`
- tear risk pixels: `2536`
- ghost edge ratio: `0.8689`
- thermal-depth centroid distance: `83.0278px`

## Ceiling Interpretation
- The selected strict result passes the edge threshold; reliability label describes residual visual risk.

## Visual Files
- acceptance summary: `acceptance_summary_panels/p29_s272_acceptance_summary.png`
- hard ceiling panel: `hard_ceiling_panels/p29_s272_hard_ceiling.png`
- reliability map: `reliability_maps/p29_s272_reliability.png`
- candidate panel: `candidate_panels/p29_s272_candidate_panel.png`
