# Phase29 v5 Explanation: 262_seq548

## Boundary
- Generation and raw selection use calibration files, raw RGB, raw LWIR, and raw depth only.
- MM5 aligned data is used after generation only for evaluation and ceiling reporting.

## Selected Result
- selected candidate: `p29_baseline`
- reliability: `accepted` / `83.6`
- selected edge: `1.3865px`
- baseline edge: `1.3865px`
- strict ceiling: `p29_v9_target_silhouette` / `0.7544px`
- scene class: `depth_tearing`
- selector reason: `selector v6 retained baseline; best raw improvement 0.2081; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: small-target guard is blocked in depth-tearing scenes so local depth correction can be tested; +18 more rejected)`
- reliability reason: `raw/depth evidence is consistent and selected edge is comfortably inside the threshold`

## Raw Evidence
- raw score: `17.6556`
- support pixels: `17247`
- rejected background pixels: `1788`
- tear risk pixels: `3177`
- ghost edge ratio: `0.8130`
- thermal-depth centroid distance: `133.5029px`

## Ceiling Interpretation
- The selected strict result passes the edge threshold; reliability label describes residual visual risk.

## Visual Files
- acceptance summary: `acceptance_summary_panels/p29_s262_acceptance_summary.png`
- hard ceiling panel: `hard_ceiling_panels/p29_s262_hard_ceiling.png`
- reliability map: `reliability_maps/p29_s262_reliability.png`
- candidate panel: `candidate_panels/p29_s262_candidate_panel.png`
