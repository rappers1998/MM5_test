# Teacher Residual Method Archive

This folder is a diagnostic archive, not the current acceptance method.

The teacher-residual route used MM5 aligned outputs as an offline teacher to study the residual alignment ceiling. That was useful for understanding the gap, but it violates the final generation boundary if promoted as a runtime method.

## Status

- Current research mainline: `darklight_mm5/calibration_only_method/phase29/`
- Stable visual acceptance baseline: `darklight_mm5/calibration_only_method/phase28/`
- This folder: diagnostic only
- Generated teacher-residual outputs, reports, flow `.npy`, and sample-flow `.npz` files have been removed from the active workspace.

## Historical Finding

The reusable global smooth-flow result improved the old calibration-plane baseline, but the stronger sample-flow result was known-sample and teacher-derived. It should be cited as an upper-bound study, not as a valid calibration/raw/depth-generated method.

## Reproduction

The source script is retained for audit purposes:

```powershell
python .\darklight_mm5\teacher_residual_method\run_teacher_residual.py --aligned-ids 106,104,103
```

Any regenerated outputs should be treated as temporary diagnostics and should not be mixed with Phase28 acceptance outputs or Phase29 broad-generalization reports.
