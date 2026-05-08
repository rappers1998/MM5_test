# HLS Registration/Fusion Skeleton

This directory contains the Vitis HLS implementation for the Phase25-to-laser FPGA path. It is the hardware-oriented branch, while the current offline acceptance branch is Phase29 v9 under `darklight_mm5/calibration_only_method/phase29/`.

## Files

- `laser_fusion.hpp`: types, constants, and top-level declarations.
- `laser_fusion.cpp`: DA1501A parser helper, range-bin selector, LWIR warp, and fusion implementation.
- `tb_laser_fusion.cpp`: C simulation testbench for UART parsing, range-bin selection, fallback, and frame warp/fusion.
- `run_hls.tcl`: Vitis HLS C simulation entry point.
- `run_vitis_hls.ps1`: local PowerShell helper for C simulation.
- `run_manual_clang_check.ps1`: fallback code-level check with the Vitis-bundled clang frontend.
- `run_hls_synth.tcl`: synthesis entry point for `phase25_laser_register_fuse_ip_top`.
- `run_vitis_hls_synth.ps1`: synthesis helper with configurable part, clock period, and Tcl file.

## Expected Integration

1. Generate `laser_lut.h`:

```powershell
python .\peizhun_jiguang\scripts\export_laser_lut.py `
  --config .\peizhun_jiguang\config\laser_registration_params.json `
  --output-dir .\peizhun_jiguang\generated
```

2. Export only `phase25_laser_register_fuse_ip_top`.
3. Feed DA1501A UART receive bytes into the final top as a packed `uart_rx_word` plus `uart_rx_count`.
4. Assert `frame_tick` once per video frame so range age/fallback logic stays synchronized.
5. Replace nearest-neighbor LWIR sampling with bilinear sampling only after resource/timing checks pass.

The older standalone top wrappers are not part of the public HLS interface.

## C Simulation

From the repository root:

```powershell
.\peizhun_jiguang\hls\run_vitis_hls.ps1
```

If Vitis HLS C simulation is blocked by the Windows/MSYS `tee.exe`, `cat.exe`, or `sh.exe` Win32 error 5 issue, use:

```powershell
.\peizhun_jiguang\hls\run_manual_clang_check.ps1
```

Expected output:

```text
tb_laser_fusion PASS
```

## Synthesis Target

The temporary FPGA device family is `xczu15eg`. The default HLS full part is `xczu15eg-ffvb1156-2-e`, with a first-pass `10 ns` clock constraint:

```powershell
.\peizhun_jiguang\hls\run_vitis_hls_synth.ps1
```

Latest unified final-IP synthesis result for `phase25_laser_register_fuse_ip_top` on `xczu15eg-ffvb1156-2-e`:

- target clock: `10 ns`
- estimated clock: `7.300 ns`
- estimated Fmax: `136.99 MHz`
- latency: `307244-307310` cycles, about `3.072-3.073 ms` per `640x480` frame
- internal image loop II: `1`
- resources: `0` BRAM18K`, `8 DSP`, `4281 FF`, `7302 LUT`, `0 URAM`

## Notes

This is a hardware starting point, not a completed board project. Board-specific AXI4-Stream, DMA, clocking, reset, UART sampling, frame sync, and video timing wrappers should be added in the FPGA project that integrates these modules.
