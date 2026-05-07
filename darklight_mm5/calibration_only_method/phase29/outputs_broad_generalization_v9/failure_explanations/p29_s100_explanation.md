# Phase29 v5 Explanation: 100_seq382

## Boundary
- Generation and raw selection use calibration files, raw RGB, raw LWIR, and raw depth only.
- MM5 aligned data is used after generation only for evaluation and ceiling reporting.

## Selected Result
- selected candidate: `p29_baseline`
- reliability: `accepted` / `83.5`
- selected edge: `1.7430px`
- baseline edge: `1.7430px`
- strict ceiling: `p29_small_target_guard` / `1.6708px`
- scene class: `reflection_background`
- selector reason: `selector v6 retained baseline; best raw improvement 0.3057; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: raw improvement -0.0891 is below 0.6500; +18 more rejected)`
- reliability reason: `raw/depth evidence is consistent and selected edge is comfortably inside the threshold`

## Raw Evidence
- raw score: `12.6145`
- support pixels: `11935`
- rejected background pixels: `2261`
- tear risk pixels: `2325`
- ghost edge ratio: `0.6188`
- thermal-depth centroid distance: `67.9885px`

## Ceiling Interpretation
- The selected strict result passes the edge threshold; reliability label describes residual visual risk.

## Visual Files
- acceptance summary: `acceptance_summary_panels/p29_s100_acceptance_summary.png`
- hard ceiling panel: `hard_ceiling_panels/p29_s100_hard_ceiling.png`
- reliability map: `reliability_maps/p29_s100_reliability.png`
- candidate panel: `candidate_panels/p29_s100_candidate_panel.png`
