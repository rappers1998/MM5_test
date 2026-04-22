from __future__ import annotations

from ..config import load_config
from ..pipeline import generate_scene_2823_comparison, run_default_suite


def main() -> None:
    config = load_config()
    run_default_suite(config)
    generate_scene_2823_comparison(config)


if __name__ == "__main__":
    main()
