# Phase29 v5 Explanation: 050_seq332

## Boundary
- Generation and raw selection use calibration files, raw RGB, raw LWIR, and raw depth only.
- MM5 aligned data is used after generation only for evaluation and ceiling reporting.

## Selected Result
- selected candidate: `p29_v9_depth_thermal`
- reliability: `risky-pass` / `78.0`
- selected edge: `1.5848px`
- baseline edge: `5.4010px`
- strict ceiling: `p29_v9_support_gated` / `1.5722px`
- scene class: `reflection_background`
- selector reason: `selector v6 selected v9_support_gated via v9_reflection_depth_thermal; raw_score delta -1.4893; adjusted score 0.3465; selection used raw RGB/LWIR/depth support gates only`
- reliability reason: `background/reflection rejection is high`

## Raw Evidence
- raw score: `18.7738`
- support pixels: `15554`
- rejected background pixels: `6860`
- tear risk pixels: `2300`
- ghost edge ratio: `0.8682`
- thermal-depth centroid distance: `50.9767px`

## Ceiling Interpretation
- The selected strict result passes the edge threshold; reliability label describes residual visual risk.

## Visual Files
- acceptance summary: `acceptance_summary_panels/p29_s050_acceptance_summary.png`
- hard ceiling panel: `hard_ceiling_panels/p29_s050_hard_ceiling.png`
- reliability map: `reliability_maps/p29_s050_reliability.png`
- candidate panel: `candidate_panels/p29_s050_candidate_panel.png`
