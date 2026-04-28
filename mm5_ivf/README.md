# MM5 IVF

This workspace is an isolated MM5 RGB+LWIR registration/fusion experiment area.
The dataset root is fixed to:

```text
D:\a三模数据
```

The current implementation keeps a reproducible engineering line:

1. build a normalized MM5 manifest from the existing benchmark CSV;
2. export full-canvas canonical RGB/LWIR pairs;
3. export affine + elastic synthetic pairs with BusReF-style reconstructible masks;
4. run compact BusReF smoke stages: reconstructor, registration, GAF fusion;
5. run an optional B-SR PDG/IPDG smoke stage;
6. export final four-panel test results without overwriting the older output directory.

## Commands

Run from the repository root:

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

## Main Outputs

- `data/processed/manifests/manifest.jsonl`
- `data/processed/canonical_pairs/`
- `data/processed/synthetic_pairs/`
- `data/processed/splits/`
- `outputs/checkpoints/busref_*_v2/`
- `outputs/checkpoints/bsr_v2_smoke/`
- `outputs/final_test_quads_v2/`

## Notes

- The indexer repairs old mojibake CSV paths such as `D:\a涓夋ā...` to `D:\a三模数据`.
- The final exporter supports `official_homography`, `depth_guided`, and `busref_refined`.
- `busref_refined` adds a guarded ECC affine refinement after the official thermal-to-RGB homography.
- `busref_gaf` uses the compact GAF checkpoint when present and falls back to luminance fusion otherwise.
