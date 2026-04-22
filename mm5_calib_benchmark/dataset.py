from __future__ import annotations

import csv
import io
import random
from pathlib import Path


def make_splits(index_rows: list[dict[str, object]], split_dir: str | Path, seed: int = 42) -> None:
    split_root = Path(split_dir)
    split_root.mkdir(parents=True, exist_ok=True)

    existing = [str(row.get("split", "")).strip() for row in index_rows]
    if any(label in {"train", "val", "test"} for label in existing):
        rows = index_rows
    else:
        rng = random.Random(seed)
        sequence_ids = sorted({int(row["sequence"]) for row in index_rows})
        rng.shuffle(sequence_ids)
        total = len(sequence_ids)
        n_train = max(1, int(round(total * 0.6)))
        n_val = max(1, int(round(total * 0.2)))
        train_ids = set(sequence_ids[:n_train])
        val_ids = set(sequence_ids[n_train : n_train + n_val])
        rows = []
        for row in index_rows:
            seq = int(row["sequence"])
            updated = dict(row)
            updated["split"] = "train" if seq in train_ids else "val" if seq in val_ids else "test"
            rows.append(updated)

    for split_name in ("train", "val", "test"):
        lines = [str(row["sequence"]) for row in rows if str(row.get("split", "")) == split_name]
        (split_root / f"{split_name}.txt").write_text("\n".join(lines), encoding="utf-8")

    fieldnames = list(rows[0].keys()) if rows else []
    with (split_root / "index_with_splits.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
