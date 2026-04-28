# Phase25 Laser-Range FPGA Registration

This package turns the best current Phase25 result into an FPGA-oriented real-time registration and fusion design.

Temporary FPGA device family: `xczu15eg`. The current HLS full part is `xczu15eg-ffvb1156-2-e`; the algorithm, range-bin strategy, and fallback rules remain unchanged.

Phase25 reference:

| Method | RGB NCC mean/min | LWIR NCC mean/min | Note |
|---|---:|---:|---|
| Phase24 board-affine baseline | `0.9865 / 0.9724` | `0.9182 / 0.9118` | calibration-board affine only |
| Phase25 promoted | `0.9865 / 0.9724` | `0.9321 / 0.9261` | dense-depth boundary selected residual shift |
| retained bridge target | - | `0.9233 / 0.9064` | previous MM5 aligned bridge level |

## Target Change

The final target is FPGA real-time multimodal registration and fusion. The runtime cannot depend on dense depth images, OpenCV searches, or MM5 aligned images. This package keeps the Phase25 geometry idea but moves expensive work offline:

1. Offline Python estimates and exports fixed parameters.
2. The DA1501A rangefinder provides one distance value per frame or at a slower control rate.
3. FPGA logic parses the distance, selects range-bin parameters, warps LWIR into the RGB canvas, and fuses RGB/LWIR in a stream-friendly path.
4. If range is invalid, stale, or inside the DA1501A blind zone, FPGA falls back to the Phase24 board-affine baseline.

Important hardware caveat: the DA1501A document states a near blind zone of up to `10 m`. The current MM5 dark tabletop samples are roughly sub-meter to 1.5 m scenes, so DA1501A cannot reproduce Phase25 dense-depth behavior on those close samples. For close-range tabletop work, use a short-range ToF/depth sensor or treat DA1501A as invalid and fall back to Phase24.

## Directory Layout

```text
peizhun_jiguang/
|- README.md
|- config/
|  `- laser_registration_params.json
|- docs/
|  |- fpga_strategy.md
|  |- rangefinder_protocol.md
|  `- verification_plan.md
|- scripts/
|  `- export_laser_lut.py
`- hls/
   |- README.md
   |- laser_fusion.hpp
   |- laser_fusion.cpp
   |- tb_laser_fusion.cpp
   |- run_hls.tcl
   |- run_hls_synth.tcl
   |- run_vitis_hls.ps1
   |- run_vitis_hls_synth.ps1
   `- run_manual_clang_check.ps1
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
6. Use `xczu15eg-ffvb1156-2-e` for the first HLS synthesis target, then replace only the package/speed-grade suffix if the final board differs.

## Current HLS Status

- Manual clang C++ validation passes: `tb_laser_fusion PASS`.
- Vitis HLS synthesis passes on `xczu15eg-ffvb1156-2-e`.
- The final export IP is `phase25_laser_register_fuse_ip_top`.
- This single top combines DA1501A Protocol 1 receive-byte parsing, range status/age handling, Phase24 fallback, Phase25-seeded range-bin selection, fixed-point LWIR warp, and RGB/LWIR fusion.
- The older standalone IP-top wrappers have been removed from the public HLS interface so the project has only one export top.
- Latest unified-IP synthesis estimate: `7.300 ns` estimated clock, `136.99 MHz` Fmax, `307244-307310` cycles / about `3.072-3.073 ms` per `640x480` frame, internal image loop `II=1`, `0` BRAM18K, `8` DSP, `4281` FF, `7302` LUT, `0` URAM.
- Vitis HLS C simulation is still blocked in this shell by the bundled MSYS `cat.exe` Win32 error 5; the code-level fallback is `hls/run_manual_clang_check.ps1`.

## Runtime Modules

- `phase25_laser_register_fuse_ip_top`: the only final HLS top to package as an IP. It accepts packed RGB frame buffers, raw LWIR frame buffers, a packed UART byte word from the DA1501A receiver, byte count, and a frame tick. It outputs the fused RGB frame plus debug/status values for latest range, flags, and age.

For a live video pipeline, wrap `phase25_laser_register_fuse_ip_top` with board-level UART sampling, frame synchronization, reset/control registers, and VDMA or ping-pong frame buffers. The current image top is frame-buffer oriented through `m_axi`; an AXI4-Stream/DMA wrapper should provide synchronized RGB/LWIR frame buffers or a dedicated LWIR line/frame buffer before replacing the memory ports with pure streaming.

## Design Boundary

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
