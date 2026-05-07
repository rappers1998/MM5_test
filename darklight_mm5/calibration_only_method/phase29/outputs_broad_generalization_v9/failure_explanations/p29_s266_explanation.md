# Phase29 v5 Explanation: 266_seq552

## Boundary
- Generation and raw selection use calibration files, raw RGB, raw LWIR, and raw depth only.
- MM5 aligned data is used after generation only for evaluation and ceiling reporting.

## Selected Result
- selected candidate: `p29_baseline`
- reliability: `accepted` / `81.7`
- selected edge: `1.7500px`
- baseline edge: `1.7500px`
- strict ceiling: `p29_raw_shift_dx3p5_dy1p5_fill` / `1.6581px`
- scene class: `depth_tearing`
- selector reason: `selector v6 retained baseline; best raw improvement 1.5931; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: small-target guard is blocked in depth-tearing scenes so local depth correction can be tested; +18 more rejected)`
- reliability reason: `raw/depth evidence is consistent and selected edge is comfortably inside the threshold`

## Raw Evidence
- raw score: `16.0972`
- support pixels: `12773`
- rejected background pixels: `1452`
- tear risk pixels: `2620`
- ghost edge ratio: `0.7728`
- thermal-depth centroid distance: `128.8643px`

## Ceiling Interpretation
- The selected strict result passes the edge threshold; reliability label describes residual visual risk.

## Visual Files
- acceptance summary: `acceptance_summary_panels/p29_s266_acceptance_summary.png`
- hard ceiling panel: `hard_ceiling_panels/p29_s266_hard_ceiling.png`
- reliability map: `reliability_maps/p29_s266_reliability.png`
- candidate panel: `candidate_panels/p29_s266_candidate_panel.png`
