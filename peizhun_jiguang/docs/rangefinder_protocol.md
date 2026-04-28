# DA1501A Rangefinder Protocol Notes

Source documents:

- `DA1501A 微型激光测距机 - 机械规格.pdf`
- `厂家测距机说明.pdf`

## Hardware Summary

| Item | Value |
|---|---|
| Wavelength | `905 nm ± 10 nm` |
| Range capability | up to `1500 m` for large targets |
| Blind zone | up to `10 m` |
| Accuracy | `±1 m` below `300 m`, `±2 m` at or above `300 m` |
| Rate | single-shot, `1 Hz`, `2 Hz`, document also lists `5 Hz` command |
| Interface | TTL UART |
| UART format | `115200 bps`, `8N1`, low bit first |
| Supply | `3 V - 5 V`, typical `3.3 V` |
| Connector | A1002WR-S-4P, mate LNN100-157128-200-4P |
| Pins | VIN+, VIN-, TTL_RXD, TTL_TXD |

## Protocol 1

### Single Measurement Command

Send:

```text
55 AA 88 FF FF FF FF checksum
```

Checksum:

```text
byte3 + byte4 + byte5 + byte6 + byte7
```

For receive frames, the document defines:

```text
receive_checksum = byte1 + byte2 + byte3 + byte4 + byte5 + byte6 + byte7
```

Return:

```text
55 AA 88 status FF DATA_H DATA_L checksum
```

Status:

- `0`: failed, `DATA_H=FF`, `DATA_L=FF`
- `1`: success

Distance:

```text
distance_m = ((DATA_H << 8) | DATA_L) / 10.0
```

Example from the document:

```text
2000.3 m -> 20003 -> 0x4E23
```

HLS parser test example:

```text
55 AA 88 01 FF 00 7B 02 -> 12.3 m -> 12300 mm
```

Here `0x02` is the low byte of `55 + AA + 88 + 01 + FF + 00 + 7B`.

### Continuous Measurement

Command byte:

| Mode | Byte |
|---|---:|
| 1 Hz | `0x89` |
| 2 Hz | `0xA9` |
| 5 Hz | `0xB9` |
| axis/alignment mode | `0xF9` |

Frame:

```text
55 AA Freq FF FF FF FF checksum
```

### Stop Measurement

```text
55 AA 8E FF FF FF FF checksum
```

## Protocol 2

Command:

```text
55 command AA
```

| Command | Meaning |
|---|---|
| `0x02` | single measurement |
| `0x03` | 1 Hz |
| `0x06` | 2 Hz |
| `0x04` | 5 Hz |
| `0x00` | stop |
| `0x0A` | axis/alignment command |

Return:

```text
AA int_high int_low frac 55
```

Distance:

```text
distance_m = integer + frac / 100.0
```

## FPGA Parser Recommendation

Use Protocol 1 first because it includes status and checksum.

Parser states:

1. wait `0x55`
2. wait `0xAA`
3. read command/frequency byte
4. read status
5. read reserved byte
6. read `DATA_H`
7. read `DATA_L`
8. read checksum
9. validate status, checksum, blind-zone, and timeout

The current HLS parser requires the Protocol 1 receive reserved byte to be `0xFF`, matching the document tables.

Recommended validity flags:

- `range_valid`: status success and checksum ok
- `range_in_blind_zone`: distance below configured blind-zone threshold
- `range_stale`: no valid update within configured frame count
- `range_use_fallback`: invalid, blind-zone, stale, or out of configured bins

## Safety and Integration Notes

- The module emits 905 nm laser light. Avoid direct eye exposure.
- Avoid measuring highly reflective targets inside the blind zone.
- Keep the optical window clean and use a 855-955 nm high-transmission coating when adding an external window.
- The FPGA algorithm must tolerate failed measurements without visual jumps.
