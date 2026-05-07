# Phase29 v5 Explanation: 123_seq409

## Boundary
- Generation and raw selection use calibration files, raw RGB, raw LWIR, and raw depth only.
- MM5 aligned data is used after generation only for evaluation and ceiling reporting.

## Selected Result
- selected candidate: `p29_v9_target_silhouette`
- reliability: `risky-pass` / `78.0`
- selected edge: `1.2253px`
- baseline edge: `4.7836px`
- strict ceiling: `p29_v9_target_silhouette` / `1.2253px`
- scene class: `general`
- selector reason: `selector v6 selected v9_support_gated via v9_target_silhouette; raw_score delta +7.8210; adjusted score 2.3043; selection used raw RGB/LWIR/depth support gates only`
- reliability reason: `double-edge/ghost risk`

## Raw Evidence
- raw score: `21.8360`
- support pixels: `5053`
- rejected background pixels: `1416`
- tear risk pixels: `2273`
- ghost edge ratio: `0.9703`
- thermal-depth centroid distance: `100.7670px`

## Ceiling Interpretation
- The selected strict result passes the edge threshold; reliability label describes residual visual risk.

## Visual Files
- acceptance summary: `acceptance_summary_panels/p29_s123_acceptance_summary.png`
- hard ceiling panel: `hard_ceiling_panels/p29_s123_hard_ceiling.png`
- reliability map: `reliability_maps/p29_s123_reliability.png`
- candidate panel: `candidate_panels/p29_s123_candidate_panel.png`
