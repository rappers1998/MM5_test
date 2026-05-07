# Phase29 v5 Explanation: 200_seq486

## Boundary
- Generation and raw selection use calibration files, raw RGB, raw LWIR, and raw depth only.
- MM5 aligned data is used after generation only for evaluation and ceiling reporting.

## Selected Result
- selected candidate: `p29_baseline`
- reliability: `risky-pass` / `68.2`
- selected edge: `2.7090px`
- baseline edge: `2.7090px`
- strict ceiling: `p29_v9_support_gated_tight` / `1.3574px`
- scene class: `double_edge_mismatch`
- selector reason: `selector v6 retained baseline; best raw improvement 0.1992; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: raw improvement 0.1992 is below 0.6500; +18 more rejected)`
- reliability reason: `double-edge/ghost risk`

## Raw Evidence
- raw score: `16.7489`
- support pixels: `5672`
- rejected background pixels: `696`
- tear risk pixels: `1314`
- ghost edge ratio: `0.9204`
- thermal-depth centroid distance: `92.3022px`

## Ceiling Interpretation
- The selected strict result passes the edge threshold; reliability label describes residual visual risk.

## Visual Files
- acceptance summary: `acceptance_summary_panels/p29_s200_acceptance_summary.png`
- hard ceiling panel: `hard_ceiling_panels/p29_s200_hard_ceiling.png`
- reliability map: `reliability_maps/p29_s200_reliability.png`
- candidate panel: `candidate_panels/p29_s200_candidate_panel.png`
