# Phase29 v5 Explanation: 137_seq423

## Boundary
- Generation and raw selection use calibration files, raw RGB, raw LWIR, and raw depth only.
- MM5 aligned data is used after generation only for evaluation and ceiling reporting.

## Selected Result
- selected candidate: `p29_baseline`
- reliability: `accepted` / `70.0`
- selected edge: `2.6714px`
- baseline edge: `2.6714px`
- strict ceiling: `p29_raw_shift_dx4p0_dy2p0_fill` / `2.5929px`
- scene class: `general`
- selector reason: `selector v6 retained baseline; best raw improvement 3.4869; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: raw improvement -0.4174 is below 0.6500; +18 more rejected)`
- reliability reason: `raw/depth evidence is consistent and selected edge is comfortably inside the threshold`

## Raw Evidence
- raw score: `13.9026`
- support pixels: `4148`
- rejected background pixels: `616`
- tear risk pixels: `1061`
- ghost edge ratio: `0.8587`
- thermal-depth centroid distance: `83.8709px`

## Ceiling Interpretation
- The selected strict result passes the edge threshold; reliability label describes residual visual risk.

## Visual Files
- acceptance summary: `acceptance_summary_panels/p29_s137_acceptance_summary.png`
- hard ceiling panel: `hard_ceiling_panels/p29_s137_hard_ceiling.png`
- reliability map: `reliability_maps/p29_s137_reliability.png`
- candidate panel: `candidate_panels/p29_s137_candidate_panel.png`
