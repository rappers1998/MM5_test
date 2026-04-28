# Documentation Index

本目录用于放项目级文档、清单和整理工具。项目入口文件仍保留在根目录：

- `README.md`
- `task_plan.md`
- `findings.md`
- `progress.md`

## Directory Map

- `manifests/`: 批量改名和文档移动清单。
  - `rename_manifest_20260428.csv`: 输出图片、指标、报告等 artifact 的旧名到新名映射。
  - `document_reorg_manifest_20260428.csv`: 文档/报告整理移动映射。
- `tools/`: 一次性整理工具。
  - `rename_artifacts_phase30.ps1`: Phase30 artifact 短命名脚本。
- `superpowers/specs/`: 之前阶段的方法设计记录。

## Project-Local Documents

- `darklight_mm5/docs/flowcharts/`: DarkLight/MM5 方法流程图。
- `darklight_mm5/*/README.md`: 各方法目录自己的入口说明。
- `peizhun_jiguang/docs/`: FPGA、测距机协议、验证计划文档。
- `runs/reports/word/`: Word 版汇总材料。
- 输出目录内的 `reports/` 或 `docs/`: 与对应实验结果强绑定的报告和说明。
