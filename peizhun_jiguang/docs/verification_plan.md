# Verification Plan

## Goal

Prove that the FPGA-oriented laser-range registration path preserves the useful part of Phase25 while removing dense depth and OpenCV runtime dependencies.

## Golden References

Use these as quality anchors:

| Reference | LWIR NCC mean/min | Role |
|---|---:|---|
| Phase24 board-affine baseline | `0.9182 / 0.9118` | fallback target |
| Phase25 depth-assisted promoted | `0.9321 / 0.9261` | stretch target |
| retained bridge target | `0.9233 / 0.9064` | historical aligned-level target |

## Offline Checks

1. JSON validates:

```powershell
python -m json.tool .\peizhun_jiguang\config\laser_registration_params.json
```

2. LUT exporter compiles:

```powershell
python -m py_compile .\peizhun_jiguang\scripts\export_laser_lut.py
```

3. LUT exporter runs:

```powershell
python .\peizhun_jiguang\scripts\export_laser_lut.py `
  --config .\peizhun_jiguang\config\laser_registration_params.json `
  --output-dir .\peizhun_jiguang\generated
```

4. Generated LUT contains:

- fallback affine
- one or more range bins
- fixed-point coefficients
- fusion coefficients

## Fixed-Point Checks

For each range bin:

1. Convert float affine to fixed-point.
2. Convert fixed-point back to float.
3. Measure max coordinate error over representative output pixels.
4. Require coordinate error below `0.25 px` for bilinear sampling or below `0.5 px` for nearest-neighbor sampling.

## Image Checks

Use Python golden model first:

1. Run Phase24 baseline.
2. Apply the same affine parameters exported for FPGA.
3. Compare registered LWIR against Phase24 output.
4. Add range-conditioned corrections.
5. Compare against Phase25 outputs and aligned-evaluation references.

Acceptance:

- fallback mode must match Phase24 within fixed-point tolerance;
- valid range mode should not degrade below Phase24;
- after real range calibration, range mode should approach Phase25.

## FPGA/HLS Checks

Temporary target device:

```text
xczu15eg
```

The synthesis helper uses the full HLS part `xczu15eg-ffvb1156-2-e` by default because Vitis requires package and speed grade. If the final board differs, pass its exact full part with `-Part` while keeping the same `xczu15eg` device family.

Local HLS discovery on this machine:

- `vitis_hls` is not currently on the PowerShell PATH.
- Use Vitis HLS 2022.1 from:

```text
F:\Vivado\vivado2022.1\Vitis_HLS\2022.1\bin\vitis_hls.bat
```

- Or open the installed Vitis HLS command prompt shortcut, whose target is:

```text
F:\Vivado\vivado2022.1\Vitis_HLS\2022.1\bin\vitis_hls_cmd.bat
```

- Legacy Vivado HLS 2018.3 also exists at:

```text
E:\vivado2018.3\vivado2018.3\Vivado\2018.3\bin\vivado_hls.bat
```

The `Vitis_HLS 2022.2` Start Menu shortcut currently points to a missing `E:\vivado1\...` target and should not be used unless that install is restored.

In this Codex shell, direct `vitis_hls.bat -version` launches the wrapper but hits a Xilinx `tee.exe` Win32 error 5. For synthesis validation, prefer the Vitis HLS command prompt shortcut or a normal user shell with the HLS environment initialized.

The helper scripts also set `XILINX_VIVADO` to:

```text
F:\Vivado\vivado2022.1\Vivado\2022.1
```

because the Vitis HLS directory alone does not contain `data\parts\arch.xml`, while the sibling Vivado install does.

By default the PowerShell launchers call `Vitis_HLS\2022.1\settings64.bat` before `loader.bat` in the same `cmd.exe` process. Use `-SkipSettings` only if the shell already has a known-good Xilinx environment.

1. C simulation with a small synthetic frame.
2. Final unified-IP synthesis check with:

```powershell
.\peizhun_jiguang\hls\run_vitis_hls_synth.ps1
```

3. C/RTL co-simulation after module interfaces are connected.
4. Resource check:
   - DSP count for affine multiply-accumulate;
   - BRAM/URAM for LWIR buffering;
   - timing at target pixel clock.
5. Hardware capture:
   - dump registered LWIR;
   - dump fused output;
   - dump range status flags.

## Current HLS Result

- Manual code-level C++ check:

```powershell
.\peizhun_jiguang\hls\run_manual_clang_check.ps1
```

Result: `tb_laser_fusion PASS`.

- Vitis HLS C simulation:
  - still blocked in this shell by Xilinx bundled MSYS `cat.exe` / Win32 error 5 before compilation;
  - use the manual clang check as the code-level C validation fallback.

- Vitis HLS synthesis for final single IP `phase25_laser_register_fuse_ip_top`:

```powershell
.\peizhun_jiguang\hls\run_vitis_hls_synth.ps1
```

Result on `xczu15eg-ffvb1156-2-e`:

| Metric | Result |
|---|---:|
| target clock | `10 ns` |
| estimated clock | `7.300 ns` |
| estimated Fmax | `136.99 MHz` |
| latency | `307244-307310 cycles / 3.072-3.073 ms` |
| internal image loop II | `1` |
| BRAM18K | `0` |
| DSP | `8` |
| FF | `4281` |
| LUT | `7302` |
| URAM | `0` |

## Runtime Fault Cases

Test these cases explicitly:

- no UART data;
- bad checksum;
- status failure;
- distance below blind zone;
- distance out of configured bins;
- stale range value;
- sudden range jump.

Expected behavior: no crash, no unbounded output jump, fallback or hold-last-good according to config.
