# Phase29 v5 Explanation: 103_seq385

## Boundary
- Generation and raw selection use calibration files, raw RGB, raw LWIR, and raw depth only.
- MM5 aligned data is used after generation only for evaluation and ceiling reporting.

## Selected Result
- selected candidate: `p29_v9_target_only`
- reliability: `risky-pass` / `78.0`
- selected edge: `1.8926px`
- baseline edge: `2.7135px`
- strict ceiling: `p29_v9_support_gated_tight` / `1.6368px`
- scene class: `general`
- selector reason: `selector v6 selected v9_support_gated via v9_weak_target_only; raw_score delta +23.7090; adjusted score 1.1953; selection used raw RGB/LWIR/depth support gates only`
- reliability reason: `double-edge/ghost risk`

## Raw Evidence
- raw score: `41.3399`
- support pixels: `5622`
- rejected background pixels: `2020`
- tear risk pixels: `1769`
- ghost edge ratio: `0.9507`
- thermal-depth centroid distance: `107.2650px`

## Ceiling Interpretation
- The selected strict result passes the edge threshold; reliability label describes residual visual risk.

## Visual Files
- acceptance summary: `acceptance_summary_panels/p29_s103_acceptance_summary.png`
- hard ceiling panel: `hard_ceiling_panels/p29_s103_hard_ceiling.png`
- reliability map: `reliability_maps/p29_s103_reliability.png`
- candidate panel: `candidate_panels/p29_s103_candidate_panel.png`
