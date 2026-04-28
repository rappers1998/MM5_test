# Teacher-Guided Residual Alignment Design

## Goal
Make the RAW RGB1 + RAW LWIR calibration result approach the precision of the current official-aligned reference path.

The final runtime path must not consume official aligned images as inputs. Official aligned RGB1/LWIR outputs may be used offline as teacher signals for diagnosis, parameter selection, and training a constrained residual correction.

## Current Baseline
- Current runtime method: `darklight_mm5/run_calibration_plane.py`
- Current output: `darklight_mm5/outputs_calibration_plane_opt`
- Teacher/reference output: `darklight_mm5/outputs`
- Current optimized mean metrics on the three dark samples:
  - `target_ncc = 0.2624`
  - `target_edge_distance = 5.4258 px`
  - `valid_ratio = 0.8337`
  - `raw_rgb_ncc = 0.2632`
  - `raw_rgb_edge_distance = 2.1906 px`
- Weak samples:
  - `104_seq386`: lowest `target_ncc = 0.1687`
  - `106_seq388`: highest `target_edge_distance = 6.6021 px`

The latest plane-parameter search gave only a small gain, so the next improvement must come from residual correction rather than another unconstrained plane sweep.

## Design Choice
Use approach C: calibration-plane initialization plus teacher-guided residual correction and constrained RAW-only refinement.

This keeps the calibrated geometry as the main physical prior, uses official aligned outputs only to learn or select a small correction, and applies guardrails so the result cannot overfit one sample or visually distort fruit/table geometry.

## Architecture

### 1. Oracle/Teacher Builder
Input:
- `index_with_splits.csv`
- `raw_rgb1_path`
- `raw_thermal16_path`
- official/reference output from `darklight_mm5/outputs`

Output:
- per-sample teacher LWIR-on-RGB images
- teacher valid masks
- teacher metrics and visual panels

Purpose:
- Define the precision target.
- Build a larger evaluation set beyond the three darkest samples.
- Keep teacher use explicit and isolated.

### 2. Residual Diagnosis
Input:
- current calibration-plane output
- teacher output
- raw RGB1/LWIR images
- annotation masks when available

Output:
- residual vector fields or landmark shifts
- global affine/homography residual estimates
- boundary error maps
- ranked failure cases

Purpose:
- Separate global residual error from local deformation.
- Identify whether a simple residual homography is enough before adding a more flexible model.

### 3. Residual Model
Start with the lowest freedom model that can beat the current baseline:

1. Global residual affine or homography.
2. If needed, low-resolution mesh or TPS with strong smoothness.
3. Reject high-frequency or nonphysical warps.

The model is trained or selected offline with teacher signals, then saved as a config/model artifact. Runtime uses only RAW images, calibration files, and the saved residual rule.

### 4. RAW-Only Refinement
After applying the learned residual model, allow a small per-sample correction using RAW-only signals:
- ROI-weighted NCC/MI
- fruit/table/turntable boundary consistency
- bounded residual shift or low-degree residual

Guardrails:
- do not reduce `valid_ratio` below `0.80`
- reject visually implausible deformation
- reject candidates that improve one sample while degrading the mean or worst sample

### 5. Evaluation and Selection
Every candidate must be evaluated against:
- teacher target metrics
- RAW RGB/LWIR metrics
- boundary diagnostics
- visual panels for all samples

The final selected candidate must improve both mean and worst-case behavior.

## Implementation Phases

### Phase A: Expand Evaluation Dataset
- Select more samples from `test,val`, including dark, medium, and difficult cases.
- Keep the original three dark samples as the core acceptance set.
- Generate a manifest with sample IDs, brightness, split, category, and all input paths.

Success:
- evaluation manifest exists
- teacher/reference outputs are found for every selected sample
- missing reference cases are reported, not silently skipped

### Phase B: Teacher Alignment Diagnostics
- Compare current `outputs_calibration_plane_opt` against teacher output.
- Estimate per-sample best residual affine/homography to measure the easy global gap.
- Produce residual panels and CSV summaries.

Success:
- report identifies whether global residual correction can explain most error
- worst samples `104_seq386` and `106_seq388` have clear failure descriptions

### Phase C: Global Residual Correction
- Add an offline search/training script for one residual affine/homography correction.
- Test fixed global correction, per-sequence correction, and per-sample oracle correction as upper/lower bounds.
- Save selected fixed correction into a new config.

Success:
- mean `target_ncc > 0.35`
- mean `target_edge_distance < 4.5 px`
- `valid_ratio >= 0.80`
- no worse visual alignment on the three dark samples

### Phase D: Low-Freedom Local Residual
- Only start this if Phase C cannot reach target.
- Use a sparse mesh or TPS with strong smoothness and small displacement limits.
- Fit on teacher residuals, then test on held-out samples.

Success:
- improves worst-case samples without creating warped fruit/table shapes
- held-out performance does not collapse

### Phase E: RAW-Only Runtime Refinement
- Add bounded per-sample refinement around the learned correction.
- Score with ROI-weighted RAW metrics and boundary metrics.
- Keep teacher metrics only for offline evaluation.

Success:
- runtime does not require official aligned images
- RAW-only refinement improves or preserves teacher metrics
- failure cases are reported with panels

### Phase F: Final Report and Promotion
- Generate final side-by-side panels:
  - official-reference teacher
  - current plane optimized
  - teacher-guided residual result
  - RAW-only refined result
- Update `README.md` and config.
- Mark the selected result directory as the recommended output.

Final acceptance:
- `target_ncc > 0.45` if feasible
- `target_edge_distance < 4.0 px` if feasible
- `valid_ratio >= 0.80`
- the weakest sample improves materially
- visual review looks close to official-reference and does not show local distortion

## Output Layout
- `darklight_mm5/outputs_teacher_residual_diagnostics`
- `darklight_mm5/outputs_teacher_residual_global`
- `darklight_mm5/outputs_teacher_residual_local`
- `darklight_mm5/outputs_teacher_residual_final`
- `darklight_mm5/teacher_residual_config.json`
- `darklight_mm5/teacher_residual_method/reports/teacher_residual_report.md`

## Risks
- Official aligned outputs can accidentally become runtime inputs. Keep teacher paths isolated in offline scripts only.
- A flexible residual warp can overfit the three dark samples. Use more samples and held-out validation.
- Metrics can improve while visual shape worsens. Visual panels remain part of acceptance.
- Thermal edges are noisy. Boundary metrics must use ROI and valid masks, not full-frame noise.

## Non-Goals
- Do not revive the rejected depth-remap implementation as the main path.
- Do not remove the current `outputs_calibration_plane_opt` baseline.
- Do not tune fusion brightness as a substitute for better registration.
