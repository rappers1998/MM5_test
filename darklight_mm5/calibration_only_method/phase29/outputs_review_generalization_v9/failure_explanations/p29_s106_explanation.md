# Phase29 v5 Explanation: 106_seq388

## Boundary
- Generation and raw selection use calibration files, raw RGB, raw LWIR, and raw depth only.
- MM5 aligned data is used after generation only for evaluation and ceiling reporting.

## Selected Result
- selected candidate: `p29_baseline`
- reliability: `accepted` / `85.8`
- selected edge: `1.4009px`
- baseline edge: `1.4009px`
- strict ceiling: `p29_raw_shift_dx2p5_dy2p0_fill` / `1.2276px`
- scene class: `general`
- selector reason: `selector v6 retained baseline; best raw improvement 0.2206; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: raw improvement -0.0417 is below 0.6500; +18 more rejected)`
- reliability reason: `raw/depth evidence is consistent and selected edge is comfortably inside the threshold`

## Raw Evidence
- raw score: `12.7881`
- support pixels: `7467`
- rejected background pixels: `1152`
- tear risk pixels: `2190`
- ghost edge ratio: `0.8186`
- thermal-depth centroid distance: `77.6728px`

## Ceiling Interpretation
- The selected strict result passes the edge threshold; reliability label describes residual visual risk.

## Visual Files
- acceptance summary: `acceptance_summary_panels/p29_s106_acceptance_summary.png`
- hard ceiling panel: `hard_ceiling_panels/p29_s106_hard_ceiling.png`
- reliability map: `reliability_maps/p29_s106_reliability.png`
- candidate panel: `candidate_panels/p29_s106_candidate_panel.png`
