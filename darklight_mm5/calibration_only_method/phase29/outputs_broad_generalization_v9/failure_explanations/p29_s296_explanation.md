# Phase29 v5 Explanation: 296_seq582

## Boundary
- Generation and raw selection use calibration files, raw RGB, raw LWIR, and raw depth only.
- MM5 aligned data is used after generation only for evaluation and ceiling reporting.

## Selected Result
- selected candidate: `p29_raw_shift_dx4p5_dy1p0_fill`
- reliability: `risky-pass` / `63.7`
- selected edge: `2.7444px`
- baseline edge: `3.1907px`
- strict ceiling: `p29_v9_depth_thermal` / `0.8395px`
- scene class: `weak_target_support`
- selector reason: `selector v6 selected raw_shift_fill via generic_v2; raw_score delta -0.2443; adjusted score 14.7023; selection used raw RGB/LWIR/depth support gates only`
- reliability reason: `weak target support`

## Raw Evidence
- raw score: `14.6023`
- support pixels: `2590`
- rejected background pixels: `584`
- tear risk pixels: `1089`
- ghost edge ratio: `0.8721`
- thermal-depth centroid distance: `61.1582px`

## Ceiling Interpretation
- The selected strict result passes the edge threshold; reliability label describes residual visual risk.

## Visual Files
- acceptance summary: `acceptance_summary_panels/p29_s296_acceptance_summary.png`
- hard ceiling panel: `hard_ceiling_panels/p29_s296_hard_ceiling.png`
- reliability map: `reliability_maps/p29_s296_reliability.png`
- candidate panel: `candidate_panels/p29_s296_candidate_panel.png`
