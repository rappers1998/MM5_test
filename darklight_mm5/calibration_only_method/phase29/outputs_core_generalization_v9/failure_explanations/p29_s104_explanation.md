# Phase29 v5 Explanation: 104_seq386

## Boundary
- Generation and raw selection use calibration files, raw RGB, raw LWIR, and raw depth only.
- MM5 aligned data is used after generation only for evaluation and ceiling reporting.

## Selected Result
- selected candidate: `p29_v9_target_only`
- reliability: `risky-pass` / `78.0`
- selected edge: `1.8814px`
- baseline edge: `2.1912px`
- strict ceiling: `p29_raw_shift_dx2p5_dy3p0_fill` / `1.4487px`
- scene class: `general`
- selector reason: `selector v6 selected v9_support_gated via v9_weak_target_only; raw_score delta +15.1334; adjusted score 1.1011; selection used raw RGB/LWIR/depth support gates only`
- reliability reason: `double-edge/ghost risk`

## Raw Evidence
- raw score: `28.8596`
- support pixels: `5836`
- rejected background pixels: `2083`
- tear risk pixels: `1842`
- ghost edge ratio: `0.9259`
- thermal-depth centroid distance: `70.1656px`

## Ceiling Interpretation
- The selected strict result passes the edge threshold; reliability label describes residual visual risk.

## Visual Files
- acceptance summary: `acceptance_summary_panels/p29_s104_acceptance_summary.png`
- hard ceiling panel: `hard_ceiling_panels/p29_s104_hard_ceiling.png`
- reliability map: `reliability_maps/p29_s104_reliability.png`
- candidate panel: `candidate_panels/p29_s104_candidate_panel.png`
