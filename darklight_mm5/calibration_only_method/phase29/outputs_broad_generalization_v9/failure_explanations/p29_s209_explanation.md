# Phase29 v5 Explanation: 209_seq495

## Boundary
- Generation and raw selection use calibration files, raw RGB, raw LWIR, and raw depth only.
- MM5 aligned data is used after generation only for evaluation and ceiling reporting.

## Selected Result
- selected candidate: `p29_raw_shift_dx4p5_dy3p0_fill`
- reliability: `risky-pass` / `64.5`
- selected edge: `2.8534px`
- baseline edge: `3.4410px`
- strict ceiling: `p29_v9_depth_thermal` / `1.4001px`
- scene class: `general`
- selector reason: `selector v6 selected raw_shift_fill via generic_v2; raw_score delta -0.6777; adjusted score 17.0605; selection used raw RGB/LWIR/depth support gates only`
- reliability reason: `edge close to threshold`

## Raw Evidence
- raw score: `16.9605`
- support pixels: `6789`
- rejected background pixels: `1068`
- tear risk pixels: `2076`
- ghost edge ratio: `0.8724`
- thermal-depth centroid distance: `58.2414px`

## Ceiling Interpretation
- The selected strict result passes the edge threshold; reliability label describes residual visual risk.

## Visual Files
- acceptance summary: `acceptance_summary_panels/p29_s209_acceptance_summary.png`
- hard ceiling panel: `hard_ceiling_panels/p29_s209_hard_ceiling.png`
- reliability map: `reliability_maps/p29_s209_reliability.png`
- candidate panel: `candidate_panels/p29_s209_candidate_panel.png`
