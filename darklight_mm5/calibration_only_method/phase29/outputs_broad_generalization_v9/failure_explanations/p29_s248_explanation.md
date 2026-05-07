# Phase29 v5 Explanation: 248_seq534

## Boundary
- Generation and raw selection use calibration files, raw RGB, raw LWIR, and raw depth only.
- MM5 aligned data is used after generation only for evaluation and ceiling reporting.

## Selected Result
- selected candidate: `p29_baseline`
- reliability: `risky-pass` / `63.1`
- selected edge: `2.7664px`
- baseline edge: `2.7664px`
- strict ceiling: `p29_v9_support_gated` / `1.4168px`
- scene class: `reflection_background`
- selector reason: `selector v6 retained baseline; best raw improvement 1.9245; safety filters rejected candidates (p29_depth_conservative: depth/reflection/edge guard candidates are diagnostic-only by default; p29_reflection_guard: depth/reflection/edge guard candidates are diagnostic-only by default; p29_small_target_guard: raw symmetric edge improvement -2.8093 is below 0.0350; +18 more rejected)`
- reliability reason: `edge close to threshold; tear-risk concentration`

## Raw Evidence
- raw score: `19.3239`
- support pixels: `13661`
- rejected background pixels: `2261`
- tear risk pixels: `3701`
- ghost edge ratio: `0.8577`
- thermal-depth centroid distance: `80.6663px`

## Ceiling Interpretation
- The selected strict result passes the edge threshold; reliability label describes residual visual risk.

## Visual Files
- acceptance summary: `acceptance_summary_panels/p29_s248_acceptance_summary.png`
- hard ceiling panel: `hard_ceiling_panels/p29_s248_hard_ceiling.png`
- reliability map: `reliability_maps/p29_s248_reliability.png`
- candidate panel: `candidate_panels/p29_s248_candidate_panel.png`
