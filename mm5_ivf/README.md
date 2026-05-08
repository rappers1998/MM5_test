# MM5 IVF

This directory is a legacy MM5 RGB/LWIR registration and fusion exploration workspace.

It is not the current acceptance route. The current acceptance route is:

```text
darklight_mm5/calibration_only_method/phase29/
```

## Status

The old generated processed data and output folders were cleaned from the active workspace. Source/config files are retained so the experiment line can be reproduced if needed.

## Original Workflow

The historical workflow was:

1. build a normalized MM5 manifest from the benchmark CSV;
2. export canonical RGB/LWIR pairs;
3. export affine and elastic synthetic pairs;
4. run compact BusReF smoke stages;
5. run optional B-SR PDG/IPDG smoke stages;
6. export four-panel test results.

## Reproduction Commands

Run from the repository root if you intentionally want to recreate the legacy generated data:

```powershell
python -m mm5_ivf.src.datasets.mm5_indexer --config mm5_ivf/configs/data_mm5.yaml
python -m mm5_ivf.src.datasets.build_canonical_pairs --config mm5_ivf/configs/data_mm5.yaml
python -m mm5_ivf.src.datasets.build_synthetic_pairs --config mm5_ivf/configs/data_mm5.yaml
python -m mm5_ivf.src.trainers.train_busref_recon --config mm5_ivf/configs/busref_mm5_v2.yaml
python -m mm5_ivf.src.trainers.train_busref_reg --config mm5_ivf/configs/busref_mm5_v2.yaml
python -m mm5_ivf.src.trainers.train_busref_fuse --config mm5_ivf/configs/busref_mm5_v2.yaml
python -m mm5_ivf.src.trainers.train_bsr_smoke --config mm5_ivf/configs/bsr_mm5_v2.yaml
python -m mm5_ivf.src.eval.export_final_test_quads --data-config mm5_ivf/configs/data_mm5.yaml --method busref_refined --fusion busref_gaf --output-dir mm5_ivf/outputs/final_test_quads_v2
```

## Notes

- Dataset root was historically fixed to `D:\a三模数据`.
- The indexer repairs old mojibake CSV paths such as `D:\a涓夋ā...` to `D:\a三模数据`.
- Regenerated `data/processed/` and `outputs/` folders should be treated as local artifacts, not as current acceptance files.
