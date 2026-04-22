from __future__ import annotations

from ..config import load_config
from ..pipeline import generate_scene_2823_comparison


def main() -> None:
    generate_scene_2823_comparison(load_config())


if __name__ == "__main__":
    main()
