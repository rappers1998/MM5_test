from __future__ import annotations

from ..config import load_config
from ..dataset import make_splits
from ..pipeline import load_index_with_splits


def main() -> None:
    config = load_config()
    rows = [row.payload for row in load_index_with_splits(config)]
    make_splits(rows, split_dir=config["outputs"]["splits_dir"], seed=int(config["runtime"]["seed"]))


if __name__ == "__main__":
    main()
