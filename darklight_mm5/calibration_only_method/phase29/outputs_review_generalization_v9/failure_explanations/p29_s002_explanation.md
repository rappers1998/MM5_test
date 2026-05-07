# Phase29 v5 Explanation: 002_seq283

## Boundary
- Generation and raw selection use calibration files, raw RGB, raw LWIR, and raw depth only.
- MM5 aligned data is used after generation only for evaluation and ceiling reporting.

## Selected Result
- selected candidate: `p29_baseline`
- reliability: `accepted` / `84.1`
- selected edge: `1.2045px`
- baseline edge: `1.2045px`
- strict ceiling: `p29_raw_shift_dx2p5_dy2p0_fill` / `1.1106px`
- scene class: `general`
- selector reason: `selector v6 retained baseline; best raw improvement 0.7997; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: raw improvement 0.2051 is below 0.6500; +18 more rejected)`
- reliability reason: `raw/depth evidence is consistent and selected edge is comfortably inside the threshold`

## Raw Evidence
- raw score: `14.5669`
- support pixels: `8877`
- rejected background pixels: `1463`
- tear risk pixels: `2928`
- ghost edge ratio: `0.8495`
- thermal-depth centroid distance: `63.4195px`

## Ceiling Interpretation
- The selected strict result passes the edge threshold; reliability label describes residual visual risk.

## Visual Files
- acceptance summary: `acceptance_summary_panels/p29_s002_acceptance_summary.png`
- hard ceiling panel: `hard_ceiling_panels/p29_s002_hard_ceiling.png`
- reliability map: `reliability_maps/p29_s002_reliability.png`
- candidate panel: `candidate_panels/p29_s002_candidate_panel.png`
