# FPGA Strategy: Phase25 to Laser-Range Runtime

## Why Phase25 Cannot Be Copied Directly

Phase25 gets its final gain by selecting a shared residual translation `dx=-2, dy=+2` from a dense depth foreground boundary. This is strong offline evidence, but it is not a direct FPGA runtime plan:

- dense depth is a full image;
- foreground boundary extraction needs morphology and connected components;
- OpenCV-style scoring is not cheap in a low-latency stream;
- a single-point laser rangefinder returns one distance, not a boundary.

The FPGA design therefore keeps Phase25's geometry stack and replaces the dense-depth decision with a range-conditioned parameter lookup.

Current temporary FPGA device family: `xczu15eg`. The first HLS full part is `xczu15eg-ffvb1156-2-e`. This affects the HLS synthesis scripts and resource/timing checks only; the registration strategy and range-bin parameters are unchanged.

## Retained From Phase25

The following are kept as the baseline geometry:

- Phase21 fixed RGB canvas: output size `640x480`, RGB crop origin around `(296,103)`.
- Phase23 LWIR crop origin: `(280,115)`.
- Phase24 board-derived affine transform:

```text
[[ 0.9642426903,  0.1313738624, -26.9919975488],
 [ -0.0663103353, 0.9894192702,  24.9203228653]]
```

- Phase25 reference residual: `dx=-2 px`, `dy=+2 px`.

For hardware inverse mapping from output pixel to raw LWIR pixel, the config stores an output-to-raw-LWIR affine seeded from the Phase24 inverse plus the Phase25 residual.

## Runtime Dataflow

```text
DA1501A TTL bytes
  -> UART frame parser
  -> distance_mm + valid/stale flags
  -> range-bin parameter select
  -> fixed-point LWIR inverse affine
  -> raw LWIR sampler
  -> registered LWIR
  -> RGB/LWIR fusion
  -> fused pixel stream
```

## Final Single HLS IP

The final exported HLS IP should be only one top:

- `phase25_laser_register_fuse_ip_top`: receives packed RGB frame buffers, raw LWIR frame buffers, packed DA1501A UART receive bytes, byte count, and frame tick. It validates Protocol 1 receive frames, tracks range age/status, selects Phase24 fallback or Phase25-seeded range-bin parameters, warps LWIR into the RGB canvas, and writes the fused RGB output frame.

The old standalone top wrappers were removed from the public HLS interface after the unified top was added. Internal helper functions remain only where `phase25_laser_register_fuse_ip_top` calls them.

Current `xczu15eg-ffvb1156-2-e` synthesis status:

| IP top | Estimated clock | Fmax | Latency | II/resources |
|---|---:|---:|---:|---|
| `phase25_laser_register_fuse_ip_top` | `7.300 ns` | `136.99 MHz` | `307244-307310 cycles / 3.072-3.073 ms` per frame | internal image loop `II=1`, `0` BRAM18K, `8` DSP, `4281` FF, `7302` LUT |

## Parameter Model

Each range bin contains:

- `min_distance_mm`, `max_distance_mm`
- output-to-raw-LWIR affine matrix in float form for offline readability
- fixed-point quantized version exported by `scripts/export_laser_lut.py`
- residual shift seed
- fusion weights

If the laser reading is invalid, stale, or below the blind-zone threshold, the parameter selector uses the fallback Phase24 baseline.

## FPGA-Friendly Choices

### Affine First

Use a 2x3 affine matrix before considering a full homography. Affine needs six multiply-accumulate operations per output pixel and is much easier to pipeline.

### Fixed-Point Format

The default exporter uses:

- affine coefficients: `Q16` fractional bits
- alpha/fusion coefficients: `Q8` fractional bits
- coordinates: signed 24-bit or wider depending on the target FPGA

Tune this after fixed-point error testing.

### Sampling

First hardware target:

- nearest-neighbor LWIR sampling for lowest resource cost;
- optional bilinear sampling when timing and BRAM budget allow.

### Fusion

Start with a simple formula:

```text
fused = ((256 - alpha) * rgb_luma + alpha * lwir_u8) >> 8
```

Use a range-bin alpha or a thresholded thermal-alpha rule. Avoid histogram equalization and large filters in the first FPGA version.

## Video Wrapper Plan

The current registration IP uses AXI master frame buffers because affine LWIR warping needs non-sequential reads. For the final real-time video system on FPGA:

1. Convert incoming RGB and LWIR AXI4-Stream video to synchronized frame buffers through VDMA or a custom ping-pong buffer.
2. Feed DA1501A receive bytes into `phase25_laser_register_fuse_ip_top` as a packed UART word/count and assert `frame_tick` at frame boundaries.
3. Run `phase25_laser_register_fuse_ip_top` once per frame; it owns range parsing, range age/fallback, registration, and fusion in one IP.
4. Stream `fused_out` back to the video pipeline through VDMA or a small output DMA.
5. Only replace frame buffers with pure streaming after adding a LWIR line/frame cache that can satisfy affine random reads.

## Accuracy Expectation

- Minimum acceptable FPGA-oriented target: match or exceed Phase24 `0.9182 / 0.9118` LWIR NCC mean/min.
- Stretch target: approach Phase25 `0.9321 / 0.9261` after real range-bin calibration.
- Do not claim Phase25 dense-depth performance from DA1501A until range-conditioned calibration is measured.

## Integration Risks

1. DA1501A blind zone is up to `10 m`; close tabletop scenes are invalid for this device.
2. One range value only constrains global scene depth, not object boundaries.
3. Laser axis must be mechanically calibrated to the RGB/LWIR camera rig.
4. If the laser spot hits background while the object of interest is foreground, the selected range bin may degrade registration.
5. Frame synchronization matters: stale range data should trigger fallback or hold-last-good with a short timeout.
