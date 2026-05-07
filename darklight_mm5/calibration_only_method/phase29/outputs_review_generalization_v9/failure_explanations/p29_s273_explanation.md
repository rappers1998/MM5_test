# Phase29 v5 Explanation: 273_seq559

## Boundary
- Generation and raw selection use calibration files, raw RGB, raw LWIR, and raw depth only.
- MM5 aligned data is used after generation only for evaluation and ceiling reporting.

## Selected Result
- selected candidate: `p29_v9_target_only`
- reliability: `risky-pass` / `78.0`
- selected edge: `1.3594px`
- baseline edge: `2.2384px`
- strict ceiling: `p29_v9_support_gated_tight` / `1.1014px`
- scene class: `general`
- selector reason: `selector v6 selected v9_support_gated via v9_weak_target_only; raw_score delta -0.6146; adjusted score 1.7722; selection used raw RGB/LWIR/depth support gates only`
- reliability reason: `support is fragmented`

## Raw Evidence
- raw score: `15.5039`
- support pixels: `4894`
- rejected background pixels: `2776`
- tear risk pixels: `2766`
- ghost edge ratio: `0.8415`
- thermal-depth centroid distance: `99.3413px`

## Ceiling Interpretation
- The selected strict result passes the edge threshold; reliability label describes residual visual risk.

## Visual Files
- acceptance summary: `acceptance_summary_panels/p29_s273_acceptance_summary.png`
- hard ceiling panel: `hard_ceiling_panels/p29_s273_hard_ceiling.png`
- reliability map: `reliability_maps/p29_s273_reliability.png`
- candidate panel: `candidate_panels/p29_s273_candidate_panel.png`
