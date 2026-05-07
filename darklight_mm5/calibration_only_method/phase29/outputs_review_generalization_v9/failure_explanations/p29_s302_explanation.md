# Phase29 v5 Explanation: 302_seq588

## Boundary
- Generation and raw selection use calibration files, raw RGB, raw LWIR, and raw depth only.
- MM5 aligned data is used after generation only for evaluation and ceiling reporting.

## Selected Result
- selected candidate: `p29_v9_target_only`
- reliability: `risky-pass` / `78.0`
- selected edge: `1.4846px`
- baseline edge: `2.5896px`
- strict ceiling: `p29_v9_support_gated` / `1.1187px`
- scene class: `general`
- selector reason: `selector v6 selected v9_support_gated via v9_weak_target_only; raw_score delta +6.8031; adjusted score 1.2427; selection used raw RGB/LWIR/depth support gates only`
- reliability reason: `double-edge/ghost risk`

## Raw Evidence
- raw score: `20.4435`
- support pixels: `6943`
- rejected background pixels: `2560`
- tear risk pixels: `1868`
- ghost edge ratio: `0.9116`
- thermal-depth centroid distance: `67.3113px`

## Ceiling Interpretation
- The selected strict result passes the edge threshold; reliability label describes residual visual risk.

## Visual Files
- acceptance summary: `acceptance_summary_panels/p29_s302_acceptance_summary.png`
- hard ceiling panel: `hard_ceiling_panels/p29_s302_hard_ceiling.png`
- reliability map: `reliability_maps/p29_s302_reliability.png`
- candidate panel: `candidate_panels/p29_s302_candidate_panel.png`
