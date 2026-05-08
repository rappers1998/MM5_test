# Project Structure Review - 2026-05-08

This note records the folder review and documentation cleanup performed after Phase28 helper code was integrated into Phase29.

## Review Scope

- Root README and all visible subproject README files.
- Top-level directory inventory and approximate size.
- Active Phase29 v9 output structure.
- Historical folders that remain for provenance.
- Documentation references that still pointed to Phase28 after Phase29 became self-contained.

## Current Mainline

The current active project route is:

```text
darklight_mm5/calibration_only_method/phase29/
```

Phase29 v9 is the accepted support-gated registration/evidence package:

- core: `3/3` pass, edge mean/max `1.7249 / 1.8926 px`;
- review: `7/7` pass, edge mean/max `1.6379 / 2.7155 px`;
- broad: `18/18` pass, edge mean/max `2.2408 / 2.9714 px`.

The final v9 generated outputs to keep are:

```text
darklight_mm5/calibration_only_method/phase29/outputs_core_generalization_v9/
darklight_mm5/calibration_only_method/phase29/outputs_review_generalization_v9/
darklight_mm5/calibration_only_method/phase29/outputs_broad_generalization_v9/
```

## Folder Inventory

| Folder | Role | Review Decision |
|---|---|---|
| `calibration/` | User calibration files. | Keep as shared source of truth. |
| `darklight_mm5/` | Main MM5 registration/fusion/evaluation workspace. | Keep; current entry is Phase29. |
| `docs/` | Project docs, manifests, tools, design records. | Keep; add this review note. |
| `mar_scholar_compare/` | Historical Scene 282/MAR-style comparison material. | Keep as archive, not active acceptance. |
| `mm5_calib_benchmark/` | Benchmark code plus split index. | Keep code and `splits/index_with_splits.csv`; old generated method outputs remain deleted. |
| `mm5_ivf/` | Legacy MM5 IVF exploration code. | Keep as reproducible historical branch, not active acceptance. |
| `peizhun_jiguang/` | DA1501A laser-assisted FPGA/HLS path. | Keep as hardware branch; update README to reference Phase29 v9. |
| `runs/` | Historical run/report artifacts. | Keep as archival evidence. |

## Naming Review

No real directory renames were made in this pass.

Reason: current output folder names are embedded in README files, reports, reproduction commands, and script arguments. Renaming them casually would reduce reproducibility. The safer policy is to document their meaning clearly.

Important name decisions:

- `phase29/` remains the active method folder because it preserves the research timeline.
- `outputs_core_generalization_v9/`, `outputs_review_generalization_v9/`, and `outputs_broad_generalization_v9/` remain unchanged because they map directly to validation profiles.
- `before_after_phase28_phase29/` remains inside v9 outputs as historical comparison evidence, not as an active Phase28 dependency.
- `peizhun_jiguang/` remains unchanged because scripts and docs already refer to that path.

If real renaming is needed later, create a manifest first and update all scripts, README files, reports, and reproducibility commands in one atomic phase.

## Issues Found

- Root README was structurally correct but needed richer project entry guidance, naming policy, maintenance boundary, and UTF-8 reading note.
- Several subproject README files still described Phase28 as the current offline acceptance route:
  - `peizhun_jiguang/README.md`
  - `peizhun_jiguang/hls/README.md`
  - `mm5_ivf/README.md`
  - `mar_scholar_compare/README.md`
  - `darklight_mm5/teacher_residual_method/README.md`
- The root README used `MM5_test/` as the logical repository name while the local folder is `MAR_bianyuan`; this is acceptable because the Git remote is `rappers1998/MM5_test.git`, but the README now explains the distinction.
- PowerShell without `-Encoding UTF8` can display the Chinese README as mojibake; the README now documents the safe read command.

## Actions Taken

- Rewrote root `README.md` as a richer project entry without dropping the existing Phase29, Phase25, FPGA/HLS, boundary, and environment content.
- Added explicit directory naming policy to avoid accidental path-breaking renames.
- Added current Phase29 v9 output structure and evidence map.
- Updated stale Phase28 references in subproject README files to Phase29 v9.
- Added this structure review note and linked it from `docs/README.md`.

## Verification Targets

- Root README should point active users to `darklight_mm5/calibration_only_method/phase29/`.
- No active subproject README should claim Phase28 is the current acceptance route.
- Phase28 may still appear as historical baseline text or in v9 comparison folder names.
- Phase29 v9 JSON reports should remain valid after documentation-only changes.
