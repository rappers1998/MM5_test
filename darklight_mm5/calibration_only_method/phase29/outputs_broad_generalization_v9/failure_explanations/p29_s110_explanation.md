# Phase29 v5 Explanation: 110_seq396

## Boundary
- Generation and raw selection use calibration files, raw RGB, raw LWIR, and raw depth only.
- MM5 aligned data is used after generation only for evaluation and ceiling reporting.

## Selected Result
- selected candidate: `p29_v9_target_only`
- reliability: `improved` / `74.0`
- selected edge: `2.2281px`
- baseline edge: `3.5698px`
- strict ceiling: `p29_v9_target_only` / `2.2281px`
- scene class: `general`
- selector reason: `selector v6 selected v9_support_gated via v9_weak_target_only; raw_score delta +2.1748; adjusted score 1.6327; selection used raw RGB/LWIR/depth support gates only`
- reliability reason: `raw/depth evidence is consistent and selected edge is comfortably inside the threshold`

## Raw Evidence
- raw score: `18.2129`
- support pixels: `7106`
- rejected background pixels: `2312`
- tear risk pixels: `2293`
- ghost edge ratio: `0.8741`
- thermal-depth centroid distance: `69.3360px`

## Ceiling Interpretation
- The selected strict result passes the edge threshold; reliability label describes residual visual risk.

## Visual Files
- acceptance summary: `acceptance_summary_panels/p29_s110_acceptance_summary.png`
- hard ceiling panel: `hard_ceiling_panels/p29_s110_hard_ceiling.png`
- reliability map: `reliability_maps/p29_s110_reliability.png`
- candidate panel: `candidate_panels/p29_s110_candidate_panel.png`
