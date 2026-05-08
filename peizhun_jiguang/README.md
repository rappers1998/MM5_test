# Phase25 Laser-Range FPGA Registration

This package is the FPGA/HLS translation path for the earlier Phase25 geometry seed. It is separate from the current Phase29 v9 support-gated offline acceptance package.

Current offline acceptance path:

```text
darklight_mm5/calibration_only_method/phase29/
```

Hardware path in this folder:

```text
single range value -> range-bin LUT -> fixed-point LWIR warp -> RGB/LWIR fusion
```

## Phase25 Reference

| Method | RGB NCC mean/min | LWIR NCC mean/min | Note |
|---|---:|---:|---|
| Phase24 board-affine baseline | `0.9865 / 0.9724` | `0.9182 / 0.9118` | calibration-board affine only |
| Phase25 promoted | `0.9865 / 0.9724` | `0.9321 / 0.9261` | dense-depth boundary selected residual shift |
| retained bridge target | - | `0.9233 / 0.9064` | previous MM5 aligned bridge level |

Phase29 v9 adds support-gated broad generalization, anti-ghost evidence views, contour overlays, reliability maps, and raw/depth-only selector gates on top of the calibration-only line. The HLS path still uses the simpler Phase25 seed because it is easier to convert into a deterministic fixed-point IP.

## Target

The runtime cannot depend on dense depth images, OpenCV searches, or MM5 aligned images. Expensive work is moved offline:

1. Offline Python estimates and exports fixed parameters.
2. The DA1501A rangefinder provides one distance value per frame or at a slower control rate.
3. FPGA logic parses the distance, selects range-bin parameters, warps LWIR into the RGB canvas, and fuses RGB/LWIR.
4. If range is invalid, stale, or inside the DA1501A blind zone, FPGA falls back to the Phase24 board-affine baseline.

Important hardware caveat: the DA1501A document states a near blind zone of up to `10 m`. The MM5 dark tabletop samples are roughly sub-meter to 1.5 m scenes, so DA1501A cannot reproduce Phase25 dense-depth behavior on those close samples. For close-range tabletop work, use a short-range ToF/depth sensor or treat DA1501A as invalid and fall back to Phase24.

## Directory Layout

```text
peizhun_jiguang/
|- README.md
|- config/
|  `- laser_registration_params.json
|- docs/
|- generated/
|- scripts/
`- hls/
```

## Recommended Flow

1. Fill `config/laser_registration_params.json` with real laser-to-camera calibration and range-bin corrections.
2. Run the LUT exporter:

```powershell
python .\peizhun_jiguang\scripts\export_laser_lut.py `
  --config .\peizhun_jiguang\config\laser_registration_params.json `
  --output-dir .\peizhun_jiguang\generated
```

3. Use `generated/laser_lut.json` or `generated/laser_lut.h` values in the FPGA build.
4. Export the final Vitis HLS IP from `phase25_laser_register_fuse_ip_top` in `hls/laser_fusion.cpp`.
5. Verify fixed-point output against the Phase25/Phase24 Python golden model before synthesis.

## Current HLS Status

- Manual clang C++ validation passes: `tb_laser_fusion PASS`.
- Vitis HLS synthesis passes on `xczu15eg-ffvb1156-2-e`.
- The final export IP is `phase25_laser_register_fuse_ip_top`.
- Latest unified-IP synthesis estimate: `7.300 ns` estimated clock, `136.99 MHz` Fmax, `307244-307310` cycles, about `3.072-3.073 ms` per `640x480` frame, internal image loop `II=1`, `0` BRAM18K, `8` DSP, `4281` FF, `7302` LUT, `0` URAM.
- Vitis HLS C simulation is still blocked in this shell by the bundled MSYS `cat.exe` Win32 error 5; the code-level fallback is `hls/run_manual_clang_check.ps1`.

## Runtime Boundary

Allowed at runtime:

- DA1501A TTL range data
- fixed camera/laser calibration parameters
- precomputed LUTs and fixed-point affine parameters
- RGB/LWIR raw streams

Not allowed at runtime:

- dense depth images
- MM5 aligned images as parameter sources
- teacher residuals
- OpenCV-style frame-wide optimization
- floating-point image registration search
