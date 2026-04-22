from __future__ import annotations

from pathlib import Path
from typing import Any


METHOD_CONFIGS = {
    "m0": "m0_mm5_official.yaml",
    "m1": "m1_zhang.yaml",
    "m2": "m2_su2025.yaml",
    "m3": "m3_jay2025.yaml",
    "m4": "m4_muhovic.yaml",
    "m5": "m5_epnp.yaml",
    "m6": "m6_mar_edge_refine.yaml",
    "m7": "m7_depth_guided_selfcal.yaml",
}


def _read_text_with_fallback(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="latin-1")


def _parse_scalar(text: str) -> Any:
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        return text[1:-1]
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_simple_yaml(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in _read_text_with_fallback(path).splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = line.lstrip(" ")
        key, sep, value = stripped.partition(":")
        if not sep:
            continue
        value = value.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        current = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _parse_scalar(value)

    return root


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_method_config(method_key: str) -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    config_name = METHOD_CONFIGS[method_key]
    return _parse_simple_yaml(package_root / "configs" / "methods" / config_name)


def load_config(method_key: str | None = None) -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parent
    config = _parse_simple_yaml(package_root / "configs" / "default.yaml")

    outputs = dict(config.get("outputs", {}))
    runtime = dict(config.get("runtime", {}))
    outputs_root = Path(outputs["root"])
    if not outputs_root.is_absolute():
        outputs_root = project_root / outputs_root
    outputs["root"] = outputs_root

    index_csv = Path(outputs["index_csv"])
    if not index_csv.is_absolute():
        index_csv = project_root / index_csv
    if not index_csv.exists():
        fallback_index = outputs["splits_dir"] if isinstance(outputs.get("splits_dir"), Path) else project_root / outputs["splits_dir"]
        fallback_index = Path(fallback_index) / "index_with_splits.csv"
        if fallback_index.exists():
            index_csv = fallback_index
    outputs["index_csv"] = index_csv

    splits_dir = Path(outputs["splits_dir"])
    if not splits_dir.is_absolute():
        splits_dir = project_root / splits_dir
    outputs["splits_dir"] = splits_dir

    config["runtime"] = runtime
    config["outputs"] = outputs
    config["package_root"] = package_root
    config["project_root"] = project_root
    config["paths"] = {
        "workspace_calibration": project_root / "calibration",
    }

    if method_key is not None:
        config["method"] = load_method_config(method_key)

    return config
