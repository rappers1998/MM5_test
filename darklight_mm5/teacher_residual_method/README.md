# Teacher Residual Method

This folder contains the new teacher-guided residual alignment method.

The method keeps `darklight_mm5/outputs_calibration_plane_opt` as the calibrated-plane baseline, uses
`darklight_mm5/outputs` only as an offline teacher/reference, and writes all new artifacts under this
folder so the previous accepted method remains untouched.

## Run

```powershell
python .\darklight_mm5\teacher_residual_method\run_teacher_residual.py --aligned-ids 106,104,103
```

Main outputs:

- `outputs/diagnostics/dl_tdiag_residuals.csv`
- `outputs/global_flow/dl_tflow_eval_ref.csv`
- `outputs/global_flow/dl_tflow_eval_sum.json`
- `outputs/sample_flow_upper_bound/dl_tsample_eval_sum.json`
- `teacher_residual_config.json`

The saved config is the runtime artifact. It stores the selected residual flow and RAW-only refinement
settings. Official/reference aligned images are not needed when applying the saved config.

`sample_flow_upper_bound` is intentionally separate: it uses sample-specific residual flows learned
offline from the teacher. Treat it as a known-sample upper bound, not the generic deployment candidate.
