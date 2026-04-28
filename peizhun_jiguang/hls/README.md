# HLS Skeleton

This directory contains a Vitis HLS-style skeleton for the Phase25-to-laser FPGA path.

The code is intentionally conservative:

- fixed image geometry: `640x480` output RGB canvas, `640x512` raw LWIR input;
- fixed-point affine inverse mapping from output pixel to raw LWIR pixel;
- nearest-neighbor LWIR sampling first;
- low-cost RGB/LWIR weighted fusion;
- range parameter selection from a small LUT.

## Files

- `laser_fusion.hpp`: types, constants, and top-level declarations.
- `laser_fusion.cpp`: range-frame parser helper, range-bin selector, LWIR warp, and fusion skeleton.
- `tb_laser_fusion.cpp`: C simulation testbench for UART parsing, range-bin selection, fallback, and frame warp/fusion.
- `run_hls.tcl`: Vitis HLS C simulation entry point.
- `run_vitis_hls.ps1`: local PowerShell helper that uses the discovered Vitis HLS 2022.1 install path by default.
- `run_manual_clang_check.ps1`: fallback code-level check that compiles and runs the same testbench with the Vitis-bundled clang frontend.
- `run_hls_synth.tcl`: synthesis entry point for the final single IP, `phase25_laser_register_fuse_ip_top`, currently targeting `xczu15eg`.
- `run_vitis_hls_synth.ps1`: PowerShell helper for synthesis with configurable part, clock period, and Tcl file.

## Expected Integration

1. Generate `laser_lut.h` with:

```powershell
python .\peizhun_jiguang\scripts\export_laser_lut.py `
  --config .\peizhun_jiguang\config\laser_registration_params.json `
  --output-dir .\peizhun_jiguang\generated
```

2. Export only `phase25_laser_register_fuse_ip_top`. It owns DA1501A receive-byte updates, status/fallback policy, Phase25 seed-bin selection, warp, and fusion.
3. Feed DA1501A UART receive bytes into the final top as a packed `uart_rx_word` plus `uart_rx_count`; assert `frame_tick` once per video frame so range age/fallback logic stays synchronized.
4. Replace nearest-neighbor sampling with bilinear sampling only after resource/timing checks pass.

The older standalone top wrappers have been removed from the public HLS interface. Internal helper functions remain in `laser_fusion.cpp` only where the final top uses them.

## C Simulation

From the repository root:

```powershell
.\peizhun_jiguang\hls\run_vitis_hls.ps1
```

The helper calls:

```powershell
F:\Vivado\vivado2022.1\Vitis_HLS\2022.1\bin\loader.bat -exec vitis_hls -f .\peizhun_jiguang\hls\run_hls.tcl
```

If the direct loader path fails on a specific Windows shell, open the `Vitis HLS 2022.1 Command Prompt` shortcut and run the same Tcl file with `vitis_hls -f`.

`run_hls.tcl` is intentionally C-simulation only and does not set an FPGA part. Add `set_part` and `create_clock` in a separate synthesis Tcl after the final board or SoC part is selected.

If Vitis HLS C simulation is blocked by the Windows/MSYS `tee.exe`, `cat.exe`, or `sh.exe` Win32 error 5 issue, use the code-level fallback:

```powershell
.\peizhun_jiguang\hls\run_manual_clang_check.ps1
```

Expected output:

```text
tb_laser_fusion PASS
```

## Synthesis Targets

The temporary FPGA device family is `xczu15eg`. The default HLS full part is `xczu15eg-ffvb1156-2-e`, with a first-pass `10 ns` clock constraint:

```powershell
.\peizhun_jiguang\hls\run_vitis_hls_synth.ps1
```

If the final board uses a different `xczu15eg` package/speed grade, keep the device family unchanged and pass the full part explicitly:

```powershell
.\peizhun_jiguang\hls\run_vitis_hls_synth.ps1 -Part "xczu15eg-<package>-<speed>"
```

Latest unified final-IP synthesis result for `phase25_laser_register_fuse_ip_top` on `xczu15eg-ffvb1156-2-e`:

- target clock: `10 ns`
- estimated clock: `7.300 ns`
- estimated Fmax: `136.99 MHz`
- latency: `307244-307310` cycles, about `3.072-3.073 ms` per `640x480` frame
- internal image loop II: `1`
- resources: `0` BRAM18K, `8` DSP, `4281` FF, `7302` LUT, `0` URAM

This keeps the Phase28 packed-image `II=1` datapath while adding the DA1501A parser/status state into the same final export top.

## Notes

This is a hardware starting point, not a completed board project. Board-specific AXI4-Stream, DMA, clocking, reset, UART sampling, frame sync, and video timing wrappers should be added in the FPGA project that integrates these modules. The current registration/fusion top is frame-buffer oriented through AXI master ports; a pure streaming version needs a synchronized LWIR line/frame buffer because affine warping uses non-sequential LWIR reads.
