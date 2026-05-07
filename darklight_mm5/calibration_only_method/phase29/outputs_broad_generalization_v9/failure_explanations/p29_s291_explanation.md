# Phase29 v5 Explanation: 291_seq577

## Boundary
- Generation and raw selection use calibration files, raw RGB, raw LWIR, and raw depth only.
- MM5 aligned data is used after generation only for evaluation and ceiling reporting.

## Selected Result
- selected candidate: `p29_baseline`
- reliability: `risky-pass` / `64.0`
- selected edge: `2.7155px`
- baseline edge: `2.7155px`
- strict ceiling: `p29_v9_support_gated` / `1.2593px`
- scene class: `double_edge_mismatch`
- selector reason: `selector v6 retained baseline; best raw improvement 0.9551; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: LWIR edge growth 1.127 exceeds 1.080; +18 more rejected)`
- reliability reason: `double-edge/ghost risk`

## Raw Evidence
- raw score: `19.3910`
- support pixels: `11381`
- rejected background pixels: `1752`
- tear risk pixels: `3089`
- ghost edge ratio: `0.9105`
- thermal-depth centroid distance: `134.2147px`

## Ceiling Interpretation
- The selected strict result passes the edge threshold; reliability label describes residual visual risk.

## Visual Files
- acceptance summary: `acceptance_summary_panels/p29_s291_acceptance_summary.png`
- hard ceiling panel: `hard_ceiling_panels/p29_s291_hard_ceiling.png`
- reliability map: `reliability_maps/p29_s291_reliability.png`
- candidate panel: `candidate_panels/p29_s291_candidate_panel.png`
