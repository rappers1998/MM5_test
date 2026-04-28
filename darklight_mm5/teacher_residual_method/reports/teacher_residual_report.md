# Teacher Residual Method Report

## Current Status
The new method is isolated under `darklight_mm5/teacher_residual_method`.

Two result tracks are produced:

- `outputs/global_flow`: reusable candidate using the median teacher-learned residual flow plus RAW-only bounded refinement.
- `outputs/sample_flow_upper_bound`: known-sample upper bound using per-sample teacher-learned residual flows. This is for diagnosis and target setting, not the generic runtime candidate.

## Metrics on Core Dark Samples

| Track | target_ncc | target_edge_distance | valid_ratio | Notes |
|---|---:|---:|---:|---|
| Previous plane optimized baseline | 0.2624 | 5.4258 px | 0.8337 | Current accepted baseline |
| Global teacher-flow + RAW refinement | 0.3074 | 4.8485 px | 0.8143 | Reusable candidate; improves NCC and edge distance |
| Per-sample teacher-flow upper bound | 0.3670 | 4.4121 px | 0.8130 | Reaches first-stage target; sample-specific upper bound |

## Interpretation
The affine residual path mostly collapsed to identity, so the remaining gap is not a single global affine correction.

The global smooth residual flow improves the baseline but does not reach `target_ncc > 0.35`.

The per-sample smooth residual flow does reach the first-stage target. This confirms the route is viable, but the residual is sample-dependent and needs a better generalized model or a defensible sample-conditioned runtime rule.

## Next Work
1. Expand teacher generation beyond the three darkest samples.
2. Train/select residual flows on a broader set and validate on held-out samples.
3. Add a constrained sample-conditioned residual model if global median flow remains below target.
4. Keep visual review mandatory because the flexible flow can improve NCC while hurting RAW RGB/LWIR consistency on some samples.

## Caveat
The sample-flow upper bound reaches the first-stage teacher target, but it is not ready to promote as the runtime method. It is sample-specific and weakens RAW RGB/LWIR NCC on `106_seq388`, so it should be treated as evidence that local residual correction can work, not as the final deployable solution.
