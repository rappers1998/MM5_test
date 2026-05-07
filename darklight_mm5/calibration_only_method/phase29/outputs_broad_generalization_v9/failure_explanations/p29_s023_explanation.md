# Phase29 v5 Explanation: 023_seq305

## Boundary
- Generation and raw selection use calibration files, raw RGB, raw LWIR, and raw depth only.
- MM5 aligned data is used after generation only for evaluation and ceiling reporting.

## Selected Result
- selected candidate: `p29_baseline`
- reliability: `accepted` / `86.2`
- selected edge: `1.4078px`
- baseline edge: `1.4078px`
- strict ceiling: `p29_raw_shift_dx2p5_dy2p0_fill` / `1.3622px`
- scene class: `general`
- selector reason: `selector v6 retained baseline; best raw improvement 0.1647; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: raw improvement -0.1222 is below 0.6500; +18 more rejected)`
- reliability reason: `raw/depth evidence is consistent and selected edge is comfortably inside the threshold`

## Raw Evidence
- raw score: `13.8679`
- support pixels: `4841`
- rejected background pixels: `944`
- tear risk pixels: `1903`
- ghost edge ratio: `0.8354`
- thermal-depth centroid distance: `46.3874px`

## Ceiling Interpretation
- The selected strict result passes the edge threshold; reliability label describes residual visual risk.

## Visual Files
- acceptance summary: `acceptance_summary_panels/p29_s023_acceptance_summary.png`
- hard ceiling panel: `hard_ceiling_panels/p29_s023_hard_ceiling.png`
- reliability map: `reliability_maps/p29_s023_reliability.png`
- candidate panel: `candidate_panels/p29_s023_candidate_panel.png`
