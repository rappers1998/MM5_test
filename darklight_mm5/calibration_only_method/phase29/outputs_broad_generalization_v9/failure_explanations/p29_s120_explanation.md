# Phase29 v5 Explanation: 120_seq406

## Boundary
- Generation and raw selection use calibration files, raw RGB, raw LWIR, and raw depth only.
- MM5 aligned data is used after generation only for evaluation and ceiling reporting.

## Selected Result
- selected candidate: `p29_raw_shift_dx3p5_dy1p5_fill`
- reliability: `risky-pass` / `62.5`
- selected edge: `2.6687px`
- baseline edge: `3.1005px`
- strict ceiling: `p29_raw_shift_dx3p5_dy1p5_fill` / `2.6687px`
- scene class: `depth_tearing`
- selector reason: `selector v6 selected raw_shift_fill via depth_tearing_v2; raw_score delta -0.1308; adjusted score 24.4086; selection used raw RGB/LWIR/depth support gates only`
- reliability reason: `tear-risk concentration; support is fragmented`

## Raw Evidence
- raw score: `17.0015`
- support pixels: `14147`
- rejected background pixels: `2200`
- tear risk pixels: `4030`
- ghost edge ratio: `0.8493`
- thermal-depth centroid distance: `41.2202px`

## Ceiling Interpretation
- The selected strict result passes the edge threshold; reliability label describes residual visual risk.

## Visual Files
- acceptance summary: `acceptance_summary_panels/p29_s120_acceptance_summary.png`
- hard ceiling panel: `hard_ceiling_panels/p29_s120_hard_ceiling.png`
- reliability map: `reliability_maps/p29_s120_reliability.png`
- candidate panel: `candidate_panels/p29_s120_candidate_panel.png`
