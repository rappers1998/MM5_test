# Phase29 v5 Explanation: 187_seq473

## Boundary
- Generation and raw selection use calibration files, raw RGB, raw LWIR, and raw depth only.
- MM5 aligned data is used after generation only for evaluation and ceiling reporting.

## Selected Result
- selected candidate: `p29_v9_depth_thermal`
- reliability: `risky-pass` / `65.2`
- selected edge: `2.7905px`
- baseline edge: `3.6135px`
- strict ceiling: `p29_v9_depth_thermal` / `2.7905px`
- scene class: `double_edge_mismatch`
- selector reason: `selector v6 selected v9_support_gated via v9_depth_thermal_double_edge; raw_score delta -9.6659; adjusted score 4.2234; selection used raw RGB/LWIR/depth support gates only`
- reliability reason: `edge close to threshold`

## Raw Evidence
- raw score: `13.6033`
- support pixels: `12166`
- rejected background pixels: `2346`
- tear risk pixels: `2847`
- ghost edge ratio: `0.7627`
- thermal-depth centroid distance: `57.2516px`

## Ceiling Interpretation
- The selected strict result passes the edge threshold; reliability label describes residual visual risk.

## Visual Files
- acceptance summary: `acceptance_summary_panels/p29_s187_acceptance_summary.png`
- hard ceiling panel: `hard_ceiling_panels/p29_s187_hard_ceiling.png`
- reliability map: `reliability_maps/p29_s187_reliability.png`
- candidate panel: `candidate_panels/p29_s187_candidate_panel.png`
