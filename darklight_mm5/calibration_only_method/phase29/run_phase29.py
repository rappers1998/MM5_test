from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PHASE29_DIR = Path(__file__).resolve().parent
METHOD_DIR = PHASE29_DIR.parent
DARKLIGHT_DIR = METHOD_DIR.parent
PHASE28_DIR = METHOD_DIR / "phase28"
for path in (DARKLIGHT_DIR, METHOD_DIR, PHASE28_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diagnose_aligned_canvas import evaluate_candidate  # noqa: E402
from run_calibration_only import CandidateOutput, sample_id  # noqa: E402
from run_darklight import auto_edges, collect_fieldnames, enhance_lowlight_bgr, imwrite_unicode, normalize_u8, write_csv  # noqa: E402
from run_phase22_stereo_recalib import add_metadata_to_metrics, summarize  # noqa: E402
from run_phase25_edge_optimization import make_shift_candidate, prepare_rows  # noqa: E402
from run_phase28 import (  # noqa: E402
    PROFILE_IDS,
    add_target_metrics,
    alpha_mask_view,
    anti_ghost_fusion,
    clean_registered_lwir_view,
    contour_alignment_overlay,
    crop_depth_to_canvas,
    depth_support_view,
    edge_error_heatmap,
    feather_alpha,
    foreground_crop,
    json_default,
    make_anti_ghost_support,
    make_panel_grid,
    make_phase28_candidate,
    make_six_panel,
    metadata_offset,
    review_support_mask,
    review_thermal_target_mask,
    simple_review_fusion,
    stabilize_lwir,
    tear_ghost_diagnostics,
    thermal_blend,
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    dx_px: float
    dy_px: float
    fill_depth_border: bool
    stabilization_kernel_px: int
    support_mode: str
    max_support_area_frac: float
    anti_ghost_erode_px: int
    anti_ghost_feather_px: float


def parse_shift_deltas(text: str) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for part in str(text).replace("/", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        fields = [field.strip() for field in part.split(",") if field.strip()]
        if len(fields) != 2:
            raise ValueError(f"invalid shift delta {part!r}; use dx,dy;dx,dy")
        out.append((float(fields[0]), float(fields[1])))
    return out


def key_float(value: float) -> str:
    return str(round(float(value), 3)).replace("-", "m").replace(".", "p")


def unique_shift_deltas(deltas: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for dx, dy in deltas:
        key = (round(float(dx), 4), round(float(dy), 4))
        if key in seen:
            continue
        seen.add(key)
        out.append((float(dx), float(dy)))
    return out


def wide_safe_shift_deltas() -> list[tuple[float, float]]:
    return [
        (-2.0, -2.0),
        (-2.0, -1.5),
        (-1.5, -2.0),
        (-1.5, -1.5),
        (1.5, -2.0),
        (1.5, -1.5),
        (1.5, -1.0),
        (2.0, -2.0),
        (2.0, -1.5),
        (2.0, -1.0),
    ]


def v5_ceiling_probe_shift_deltas() -> list[tuple[float, float]]:
    return [
        (-4.0, -3.0),
        (-4.0, -2.5),
        (-4.0, -2.0),
        (-3.5, -3.0),
        (-3.5, -2.5),
        (-3.5, -2.0),
        (-3.0, -2.5),
        (-3.0, -2.0),
        (-2.5, -2.0),
        (-2.5, -1.5),
        (-2.0, -2.0),
        (-2.0, -1.5),
        (2.5, -2.5),
        (2.5, -2.0),
        (2.5, -1.5),
        (2.5, -1.0),
        (3.0, -2.5),
        (3.0, -2.0),
        (3.0, -1.5),
        (3.0, -1.0),
        (3.5, -2.5),
        (3.5, -2.0),
        (3.5, -1.5),
    ]


def v6_risk_shift_deltas() -> list[tuple[float, float]]:
    return v5_ceiling_probe_shift_deltas()


def v7_edge_micro_shift_deltas() -> list[tuple[float, float]]:
    deltas: list[tuple[float, float]] = []
    for dx in (-2.75, -2.5, -2.25, -2.0, -1.75, -1.5):
        for dy in (-2.25, -2.0, -1.75, -1.5, -1.25, -1.0):
            deltas.append((dx, dy))
    for dx in (-4.5, -4.25, -4.0, -3.75, -3.5, -3.25, -3.0, -2.75, -2.5):
        for dy in (-3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0):
            deltas.append((dx, dy))
    for dx in (-0.25, 0.0, 0.25, 0.5):
        for dy in (-1.25, -1.0, -0.75, -0.5, -0.25, 0.0):
            deltas.append((dx, dy))
    return unique_shift_deltas(deltas)


def v7_compact_edge_shift_deltas() -> list[tuple[float, float]]:
    return unique_shift_deltas(
        [
            (-4.50, -3.00),
            (-4.50, -2.75),
            (-4.50, -2.50),
            (-4.25, -3.00),
            (-4.00, -3.00),
            (-3.50, -2.50),
            (-3.00, -2.25),
            (-2.50, -1.75),
            (-2.00, -1.25),
            (-2.00, -1.00),
            (-1.50, -1.25),
            (-1.00, -1.00),
            (-0.50, -1.25),
            (0.00, -1.25),
            (0.00, -1.00),
            (0.25, -1.25),
            (0.50, -1.25),
            (0.50, -1.00),
        ]
    )


def acceptance_lite_shift_deltas() -> list[tuple[float, float]]:
    return [(0.0, -0.5), (1.0, -1.0), (1.0, 1.0)]


def acceptance_lite_risk_shift_deltas() -> list[tuple[float, float]]:
    return [(2.5, -2.5), (2.5, -1.5), (2.5, -1.0), (3.5, -2.0), (3.5, -1.5)]


def active_shift_deltas(args) -> list[tuple[float, float]]:
    if str(args.candidate_grid) == "acceptance-lite":
        return acceptance_lite_shift_deltas()
    deltas = parse_shift_deltas(args.shift_deltas)
    if str(args.candidate_grid) in {"wide-safe", "edge-v7", "edge-v7-compact"} or str(args.run_mode) == "ceiling-study":
        deltas.extend(wide_safe_shift_deltas())
    return unique_shift_deltas(deltas)


def active_v5_ceiling_probe_deltas(args) -> list[tuple[float, float]]:
    if str(getattr(args, "version_label", "")) != "v5":
        return []
    if str(args.candidate_grid) != "wide-safe" and str(args.run_mode) != "ceiling-study":
        return []
    return unique_shift_deltas(v5_ceiling_probe_shift_deltas())


def active_v6_risk_shift_deltas(args) -> list[tuple[float, float]]:
    if str(getattr(args, "version_label", "")) not in {"v6", "v7", "v8"}:
        return []
    if str(args.candidate_grid) == "acceptance-lite":
        return acceptance_lite_risk_shift_deltas()
    if str(args.candidate_grid) not in {"wide-safe", "edge-v7", "edge-v7-compact", "component-v8"} and str(args.run_mode) != "ceiling-study":
        return []
    return unique_shift_deltas(v6_risk_shift_deltas())


def active_v7_edge_micro_deltas(args) -> list[tuple[float, float]]:
    if str(getattr(args, "version_label", "")) != "v7":
        return []
    if str(args.candidate_grid) == "edge-v7-compact":
        return v7_compact_edge_shift_deltas()
    if str(args.candidate_grid) != "edge-v7" and str(args.run_mode) != "ceiling-study":
        return []
    return v7_edge_micro_shift_deltas()


def candidate_specs(args) -> list[CandidateSpec]:
    seen: set[tuple[float, float, bool, int, str]] = set()
    specs: list[CandidateSpec] = []

    def add(
        name: str,
        dx: float,
        dy: float,
        fill: bool,
        kernel: int,
        mode: str,
        max_area: float,
        erode: int,
        feather: float,
    ) -> None:
        key = (round(dx, 4), round(dy, 4), bool(fill), int(kernel), str(mode))
        if key in seen:
            return
        seen.add(key)
        specs.append(
            CandidateSpec(
                name=name,
                dx_px=float(dx),
                dy_px=float(dy),
                fill_depth_border=bool(fill),
                stabilization_kernel_px=int(kernel),
                support_mode=str(mode),
                max_support_area_frac=float(max_area),
                anti_ghost_erode_px=int(erode),
                anti_ghost_feather_px=float(feather),
            )
        )

    add(
        "p29_baseline",
        args.dx_px,
        args.dy_px,
        True,
        args.stabilization_kernel_px,
        "baseline",
        args.max_support_area_frac,
        args.anti_ghost_erode_px,
        args.anti_ghost_feather_px,
    )
    add(
        "p29_depth_conservative",
        args.dx_px,
        args.dy_px,
        False,
        args.stabilization_kernel_px,
        "depth_conservative",
        min(args.max_support_area_frac, 0.075),
        max(args.anti_ghost_erode_px, 3),
        args.anti_ghost_feather_px + 1.0,
    )
    add(
        "p29_reflection_guard",
        args.dx_px,
        args.dy_px,
        False,
        args.stabilization_kernel_px,
        "reflection_guard",
        min(args.max_support_area_frac, 0.055),
        max(args.anti_ghost_erode_px, 3),
        args.anti_ghost_feather_px + 1.5,
    )
    add(
        "p29_small_target_guard",
        args.dx_px,
        args.dy_px,
        True,
        max(5, args.stabilization_kernel_px - 4),
        "small_target_guard",
        min(args.max_support_area_frac, 0.065),
        args.anti_ghost_erode_px,
        args.anti_ghost_feather_px,
    )
    add(
        "p29_edge_contamination_guard",
        args.dx_px,
        args.dy_px,
        False,
        args.stabilization_kernel_px,
        "edge_contamination_guard",
        min(args.max_support_area_frac, 0.08),
        max(args.anti_ghost_erode_px, 3),
        args.anti_ghost_feather_px + 1.0,
    )
    if str(getattr(args, "version_label", "")) in {"v5", "v7"}:
        add(
            "p29_reflection_hard_reject",
            args.dx_px,
            args.dy_px,
            False,
            args.stabilization_kernel_px,
            "reflection_hard_reject",
            min(args.max_support_area_frac, 0.035),
            max(args.anti_ghost_erode_px, 4),
            args.anti_ghost_feather_px + 2.0,
        )
        add(
            "p29_reflection_tight_target",
            args.dx_px,
            args.dy_px,
            True,
            max(5, args.stabilization_kernel_px - 4),
            "reflection_tight_target",
            min(args.max_support_area_frac, 0.040),
            max(args.anti_ghost_erode_px, 4),
            args.anti_ghost_feather_px + 1.0,
        )
        add(
            "p29_small_target_conservative",
            args.dx_px,
            args.dy_px,
            True,
            max(5, args.stabilization_kernel_px - 6),
            "small_target_conservative",
            min(args.max_support_area_frac, 0.045),
            max(args.anti_ghost_erode_px, 3),
            args.anti_ghost_feather_px + 1.0,
        )
        add(
            "p29_foreground_tight",
            args.dx_px,
            args.dy_px,
            True,
            args.stabilization_kernel_px,
            "foreground_tight",
            min(args.max_support_area_frac, 0.055),
            max(args.anti_ghost_erode_px, 3),
            args.anti_ghost_feather_px + 1.0,
        )
        add(
            "p29_depth_invalid_reject",
            args.dx_px,
            args.dy_px,
            False,
            args.stabilization_kernel_px,
            "depth_invalid_reject",
            min(args.max_support_area_frac, 0.060),
            max(args.anti_ghost_erode_px, 3),
            args.anti_ghost_feather_px + 1.0,
        )

    if str(getattr(args, "version_label", "")) == "v7":
        for kernel in (5, 7, 11):
            add(
                f"p29_v7_kernel{kernel}_fill",
                args.dx_px,
                args.dy_px,
                True,
                kernel,
                "v7_edge_local",
                min(args.max_support_area_frac, 0.070),
                args.anti_ghost_erode_px,
                args.anti_ghost_feather_px,
            )
        add(
            "p29_v7_reflection_core",
            args.dx_px,
            args.dy_px,
            True,
            5,
            "v7_reflection_core",
            min(args.max_support_area_frac, 0.040),
            max(args.anti_ghost_erode_px, 4),
            args.anti_ghost_feather_px + 1.0,
        )
        add(
            "p29_v7_weak_target_core",
            args.dx_px,
            args.dy_px,
            True,
            5,
            "v7_weak_target_core",
            min(args.max_support_area_frac, 0.055),
            max(args.anti_ghost_erode_px, 3),
            args.anti_ghost_feather_px + 0.5,
        )

    if str(args.candidate_grid) == "component-v8" or str(getattr(args, "version_label", "")) == "v8":
        add(
            "p29_v8_depth_projection_only",
            args.dx_px,
            args.dy_px,
            False,
            max(5, args.stabilization_kernel_px - 4),
            "v8_depth_projection",
            min(args.max_support_area_frac, 0.070),
            max(args.anti_ghost_erode_px, 2),
            args.anti_ghost_feather_px,
        )
        for mode in ("v8_component_rgb", "v8_component_depth", "v8_component_mixed"):
            add(
                f"p29_{mode}",
                args.dx_px,
                args.dy_px,
                True,
                max(5, args.stabilization_kernel_px - 4),
                mode,
                min(args.max_support_area_frac, 0.070),
                max(args.anti_ghost_erode_px, 2),
                args.anti_ghost_feather_px,
            )

    if str(args.candidate_grid) == "support-v9" or str(getattr(args, "version_label", "")) == "v9":
        for name, mode, max_area, erode, feather in (
            ("p29_v9_support_gated", "v9_support_gated", 0.075, 3, 4.0),
            ("p29_v9_support_gated_tight", "v9_support_gated_tight", 0.055, 4, 5.0),
            ("p29_v9_target_only", "v9_target_only", 0.045, 4, 5.0),
            ("p29_v9_depth_thermal", "v9_depth_thermal", 0.065, 3, 4.0),
            ("p29_v9_target_silhouette", "v9_target_silhouette", 0.045, 3, 4.0),
        ):
            add(
                name,
                args.dx_px,
                args.dy_px,
                True,
                max(5, args.stabilization_kernel_px - 4),
                mode,
                min(args.max_support_area_frac, max_area),
                max(args.anti_ghost_erode_px, erode),
                args.anti_ghost_feather_px + feather,
            )

    for delta_dx, delta_dy in active_shift_deltas(args):
        if abs(delta_dx) < 1e-9 and abs(delta_dy) < 1e-9:
            continue
        dx = float(args.dx_px) + float(delta_dx)
        dy = float(args.dy_px) + float(delta_dy)
        tag = f"dx{key_float(dx)}_dy{key_float(dy)}"
        add(
            f"p29_raw_shift_{tag}_fill",
            dx,
            dy,
            True,
            args.stabilization_kernel_px,
            "raw_shift_local",
            args.max_support_area_frac,
            args.anti_ghost_erode_px,
            args.anti_ghost_feather_px,
        )
    for delta_dx, delta_dy in active_v7_edge_micro_deltas(args):
        dx = float(args.dx_px) + float(delta_dx)
        dy = float(args.dy_px) + float(delta_dy)
        tag = f"dx{key_float(dx)}_dy{key_float(dy)}"
        add(
            f"p29_v7_edge_shift_{tag}_fill",
            dx,
            dy,
            True,
            5,
            "v7_edge_local",
            min(args.max_support_area_frac, 0.065),
            args.anti_ghost_erode_px,
            args.anti_ghost_feather_px,
        )
        if bool(args.include_nofill_shift_candidates):
            add(
                f"p29_raw_shift_{tag}_nofill",
                dx,
                dy,
                False,
                args.stabilization_kernel_px,
                "raw_shift_local",
                min(args.max_support_area_frac, 0.075),
                max(args.anti_ghost_erode_px, 3),
                args.anti_ghost_feather_px + 0.5,
            )
    for delta_dx, delta_dy in active_v5_ceiling_probe_deltas(args):
        dx = float(args.dx_px) + float(delta_dx)
        dy = float(args.dy_px) + float(delta_dy)
        tag = f"dx{key_float(dx)}_dy{key_float(dy)}"
        add(
            f"p29_v5_probe_{tag}_fill",
            dx,
            dy,
            True,
            args.stabilization_kernel_px,
            "v5_ceiling_probe",
            min(args.max_support_area_frac, 0.080),
            args.anti_ghost_erode_px,
            args.anti_ghost_feather_px,
        )
    for delta_dx, delta_dy in active_v6_risk_shift_deltas(args):
        dx = float(args.dx_px) + float(delta_dx)
        dy = float(args.dy_px) + float(delta_dy)
        tag = f"dx{key_float(dx)}_dy{key_float(dy)}"
        add(
            f"p29_v6_risk_shift_{tag}_fill",
            dx,
            dy,
            True,
            args.stabilization_kernel_px,
            "v6_risk_shift",
            min(args.max_support_area_frac, 0.085),
            args.anti_ghost_erode_px,
            args.anti_ghost_feather_px,
        )
    return specs


def make_phase29_candidate(item: dict, args, spec: CandidateSpec) -> CandidateOutput:
    if spec.name == "p29_baseline":
        baseline_args = copy.copy(args)
        baseline_args.current_profile = args.current_profile
        baseline_args.no_depth_border_fill = False
        baseline_args.stabilization_kernel_px = spec.stabilization_kernel_px
        baseline_args.visual_mode = "anti-ghost"
        baseline_args.max_support_area_frac = spec.max_support_area_frac
        baseline_args.anti_ghost_erode_px = spec.anti_ghost_erode_px
        baseline_args.anti_ghost_feather_px = spec.anti_ghost_feather_px
        candidate = make_phase28_candidate(item, baseline_args)
        metadata = dict(candidate.metadata)
    else:
        extra_metadata = {}
        if spec.support_mode == "v8_depth_projection":
            base = item["depth_projection"]
            metadata = dict(base.metadata)
            extra_metadata["phase29_v8_depth_projection_only"] = True
        else:
            base = make_shift_candidate(
                item["phase24"],
                item["depth_projection"],
                spec.dx_px,
                spec.dy_px,
                fill_depth_border=spec.fill_depth_border,
            )
            metadata = dict(base.metadata)
        if str(spec.support_mode).startswith("v8_component_"):
            lwir_local, valid_local, snap_meta = make_component_snap_candidate(item, base, args, spec.support_mode)
            base = CandidateOutput(
                name=spec.name,
                rgb=base.rgb,
                lwir=lwir_local,
                rgb_valid=base.rgb_valid,
                lwir_valid=valid_local,
                metadata={**metadata, **snap_meta},
            )
        if str(spec.support_mode).startswith("v9_"):
            lwir_local, valid_local, gated_meta = make_support_gated_lwir_candidate(item, base, args, spec.support_mode)
            base = CandidateOutput(
                name=spec.name,
                rgb=base.rgb,
                lwir=lwir_local,
                rgb_valid=base.rgb_valid,
                lwir_valid=valid_local,
                metadata={**metadata, **gated_meta},
            )
        lwir = stabilize_lwir(base.lwir, spec.stabilization_kernel_px)
        metadata = dict(base.metadata)
        metadata.update(extra_metadata)
        candidate = CandidateOutput(
            name=spec.name,
            rgb=base.rgb,
            lwir=lwir,
            rgb_valid=base.rgb_valid,
            lwir_valid=base.lwir_valid,
            metadata=metadata,
        )

    metadata.update(
        {
            "method": spec.name,
            "candidate": spec.name,
            "allowed_for_generation": True,
            "phase29_class": "broad_generalization_raw_only_selection",
            "phase29_profile": args.current_profile,
            "phase29_support_mode": spec.support_mode,
            "phase29_selection_uses_aligned": False,
            "phase29_selector_version": str(args.selector_version),
            "phase29_version_label": str(getattr(args, "version_label", "")),
            "phase29_candidate_grid": str(args.candidate_grid),
            "phase29_run_mode": str(args.run_mode),
            "phase29_report_level": str(args.report_level),
            "phase29_explainability_level": str(getattr(args, "explainability_level", "")),
            "phase29_reliability_gate": str(getattr(args, "reliability_gate", "")),
            "phase29_failure_focus": str(getattr(args, "failure_focus", "")),
            "depth_registration_dx_px": float(spec.dx_px),
            "depth_registration_dy_px": float(spec.dy_px),
            "stabilization_kernel_px": int(spec.stabilization_kernel_px),
            "fill_depth_border": bool(spec.fill_depth_border),
            "uses_aligned_for_generation": False,
            "anti_ghost_feather_px": float(spec.anti_ghost_feather_px),
            "anti_ghost_erode_px": int(spec.anti_ghost_erode_px),
            "max_support_area_frac": float(spec.max_support_area_frac),
            "calibration_file": str(args.calibration),
            "thermal_camera_calibration_file": str(args.thermal_camera_calibration),
            "target_size": str(args.target_size),
            "rule_source": (
                "Phase29 candidate generated from calibration files, raw RGB/LWIR/depth, "
                "Phase24 board-affine geometry, Phase25 depth projection support, and "
                "raw-only scene reliability scoring. MM5 aligned images are evaluation-only."
            ),
        }
    )
    return CandidateOutput(
        name=spec.name,
        rgb=candidate.rgb,
        lwir=candidate.lwir,
        rgb_valid=candidate.rgb_valid,
        lwir_valid=candidate.lwir_valid,
        metadata=metadata,
    )


def edge_distance(moving_edges: np.ndarray, fixed_edges: np.ndarray, roi: np.ndarray) -> tuple[float, int]:
    moving = moving_edges.astype(bool) & roi.astype(bool)
    fixed = fixed_edges.astype(bool) & roi.astype(bool)
    moving_count = int(np.count_nonzero(moving))
    fixed_count = int(np.count_nonzero(fixed))
    if moving_count < 20 or fixed_count < 20:
        return 12.0, moving_count
    distance = cv2.distanceTransform((~fixed).astype(np.uint8) * 255, cv2.DIST_L2, 3)
    return float(np.mean(distance[moving])), moving_count


def component_centroids(mask: np.ndarray, min_area: int) -> list[dict]:
    clean = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(clean, 8)
    comps: list[dict] = []
    for idx in range(1, count):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        comps.append(
            {
                "idx": idx,
                "area": area,
                "centroid": (float(centroids[idx][0]), float(centroids[idx][1])),
                "mask": labels == idx,
            }
        )
    comps.sort(key=lambda item: int(item["area"]), reverse=True)
    return comps


def local_background_value(lwir: np.ndarray, valid: np.ndarray, mask: np.ndarray) -> int:
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((19, 19), np.uint8), iterations=1).astype(bool) & ~mask.astype(bool)
    vals = lwir[ring & valid.astype(bool)]
    if vals.size < 20:
        vals = lwir[valid.astype(bool)]
    if vals.size == 0:
        return 0
    return int(np.median(vals))


def refine_support_gate(mask: np.ndarray, valid: np.ndarray, erode_px: int, dilate_px: int, max_components: int = 4) -> np.ndarray:
    gate = mask.astype(bool) & valid.astype(bool)
    if int(np.count_nonzero(gate)) < 40:
        return gate
    if erode_px > 0:
        kernel = np.ones((max(1, int(erode_px)), max(1, int(erode_px))), np.uint8)
        gate = cv2.erode(gate.astype(np.uint8), kernel, iterations=1).astype(bool)
    gate = cv2.morphologyEx(gate.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    gate = cv2.morphologyEx(gate.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1).astype(bool)
    components = component_centroids(gate, min_area=40)
    if components:
        kept = np.zeros_like(gate, dtype=bool)
        for comp in components[:max_components]:
            kept |= comp["mask"]
        gate = kept
    if dilate_px > 0 and int(np.count_nonzero(gate)) >= 40:
        kernel = np.ones((max(1, int(dilate_px)), max(1, int(dilate_px))), np.uint8)
        gate = cv2.dilate(gate.astype(np.uint8), kernel, iterations=1).astype(bool)
    return gate & valid.astype(bool)


def make_support_gated_lwir_candidate(item: dict, base: CandidateOutput, args, mode: str) -> tuple[np.ndarray, np.ndarray, dict]:
    valid = base.rgb_valid.astype(bool) & base.lwir_valid.astype(bool)
    rgb_offset = metadata_offset(base.metadata, "rgb_crop_offset_xy")
    target_size = (base.rgb.shape[1], base.rgb.shape[0])
    support = review_support_mask(base.rgb, valid, item.get("depth"), rgb_offset, item["row"].raw_rgb_anno_class_path)
    depth_crop, depth_valid = crop_depth_to_canvas(item, rgb_offset, target_size)
    local_args = support_args(args, base)
    anti = make_anti_ghost_support(base.rgb, base.lwir, valid, support, depth_crop, depth_valid, local_args)
    anti_mask = anti.get("mask", np.zeros_like(valid)).astype(bool) & valid
    thermal = anti.get("thermal_mask", np.zeros_like(valid)).astype(bool) & valid
    depth_mask = anti.get("depth_mask", np.zeros_like(valid)).astype(bool) & valid
    support = support.astype(bool) & valid
    support_band = cv2.dilate(support.astype(np.uint8), np.ones((23, 23), np.uint8), iterations=1).astype(bool) if np.any(support) else support
    thermal_band = cv2.dilate(thermal.astype(np.uint8), np.ones((19, 19), np.uint8), iterations=1).astype(bool) if np.any(thermal) else thermal
    depth_band = cv2.dilate(depth_mask.astype(np.uint8), np.ones((17, 17), np.uint8), iterations=1).astype(bool) if np.any(depth_mask) else depth_mask

    if mode == "v9_support_gated_tight":
        gate = (anti_mask & (support_band | depth_band)) | (thermal & support_band)
        erode_px, dilate_px, feather_px = 2, 7, 7.0
    elif mode in {"v9_target_only", "v9_target_silhouette"}:
        gate = thermal & (support_band | depth_band)
        erode_px, dilate_px, feather_px = 1, 9, 8.0 if mode == "v9_target_only" else 4.0
    elif mode == "v9_depth_thermal":
        gate = (thermal & depth_band) | (anti_mask & depth_band) | (support & thermal_band)
        erode_px, dilate_px, feather_px = 1, 9, 7.0
    elif mode == "v9_support_silhouette":
        gate = (anti_mask & (support_band | depth_band)) | (support & thermal_band)
        erode_px, dilate_px, feather_px = 1, 9, 4.0
    else:
        gate = anti_mask | (thermal & support_band) | (support & thermal_band)
        erode_px, dilate_px, feather_px = 1, 11, 6.0

    if int(np.count_nonzero(gate)) < 80:
        gate = (support | thermal | depth_mask) & valid
    gate = refine_support_gate(gate, valid, erode_px, dilate_px)
    if int(np.count_nonzero(gate)) < 80:
        return base.lwir.copy(), base.lwir_valid.copy(), {"phase29_v9_support_gated": False, "phase29_v9_gate_pixels": 0}

    bg_value = local_background_value(base.lwir, valid, gate)
    smooth = cv2.GaussianBlur(base.lwir, (31, 31), 0).astype(np.float32)
    background = smooth * 0.35 + float(bg_value) * 0.65
    alpha = feather_alpha(gate, valid, feather_px).astype(np.float32)
    if mode in {"v9_target_silhouette", "v9_support_silhouette"}:
        target_vals = base.lwir[gate & valid]
        target_value = float(np.percentile(target_vals, 82)) if target_vals.size else float(bg_value)
        target_layer = np.full_like(background, target_value, dtype=np.float32)
        out = target_layer * alpha + background * (1.0 - alpha)
    else:
        target_value = float("nan")
        out = base.lwir.astype(np.float32) * alpha + background * (1.0 - alpha)
    out = np.clip(out, 0, 255).astype(np.uint8)
    out_valid = base.lwir_valid.astype(bool).copy()
    return out, out_valid, {
        "phase29_v9_support_gated": True,
        "phase29_v9_gate_mode": mode,
        "phase29_v9_gate_pixels": int(np.count_nonzero(gate)),
        "phase29_v9_background_value": int(bg_value),
        "phase29_v9_target_value": float(target_value) if np.isfinite(target_value) else "",
        "phase29_v9_valid_ratio": float(np.count_nonzero(out_valid) / max(1, out_valid.size)),
    }


def make_component_snap_candidate(item: dict, base: CandidateOutput, args, mode: str) -> tuple[np.ndarray, np.ndarray, dict]:
    valid = base.rgb_valid.astype(bool) & base.lwir_valid.astype(bool)
    rgb_offset = metadata_offset(base.metadata, "rgb_crop_offset_xy")
    target_size = (base.rgb.shape[1], base.rgb.shape[0])
    support = review_support_mask(base.rgb, valid, item.get("depth"), rgb_offset, item["row"].raw_rgb_anno_class_path)
    depth_crop, depth_valid = crop_depth_to_canvas(item, rgb_offset, target_size)
    local_args = support_args(args, base)
    anti = make_anti_ghost_support(base.rgb, base.lwir, valid, support, depth_crop, depth_valid, local_args)
    thermal = review_thermal_target_mask(base.lwir, valid, support if np.any(support) else valid)
    depth_mask = anti.get("depth_mask", np.zeros_like(valid)).astype(bool) & valid
    if mode == "v8_component_rgb":
        target = support.astype(bool) & valid
    elif mode == "v8_component_depth":
        target = depth_mask
    else:
        target = (support.astype(bool) | depth_mask) & valid
    if int(np.count_nonzero(target)) < 40 or int(np.count_nonzero(thermal)) < 40:
        return base.lwir.copy(), base.lwir_valid.copy(), {"component_snap_pairs": 0}

    thermal_components = component_centroids(thermal, int(args.v8_component_min_thermal_area))
    target_components = component_centroids(target, int(args.v8_component_min_target_area))
    if not thermal_components or not target_components:
        return base.lwir.copy(), base.lwir_valid.copy(), {"component_snap_pairs": 0}

    height, width = base.lwir.shape[:2]
    out = base.lwir.copy()
    out_valid = base.lwir_valid.astype(bool).copy()
    used_targets: set[int] = set()
    pair_count = 0
    total_shift = 0.0
    max_shift = float(args.v8_component_max_shift_px)
    paste_order: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    erase_mask = np.zeros_like(valid, dtype=bool)
    for comp in thermal_components[: int(args.v8_component_max_components)]:
        cx, cy = comp["centroid"]
        best = None
        best_dist = float("inf")
        for target_idx, target_comp in enumerate(target_components[: int(args.v8_component_max_components) + 3]):
            if target_idx in used_targets:
                continue
            tx, ty = target_comp["centroid"]
            dist = float(np.hypot(tx - cx, ty - cy))
            if dist < best_dist:
                best = (target_idx, target_comp, tx, ty)
                best_dist = dist
        if best is None or best_dist > float(args.v8_component_pair_max_distance_px):
            continue
        target_idx, _target_comp, tx, ty = best
        dx = float(np.clip(tx - cx, -max_shift, max_shift))
        dy = float(np.clip(ty - cy, -max_shift, max_shift))
        if abs(dx) + abs(dy) < float(args.v8_component_min_shift_l1):
            continue
        used_targets.add(target_idx)
        src = cv2.dilate(comp["mask"].astype(np.uint8), np.ones((int(args.v8_component_dilate_px), int(args.v8_component_dilate_px)), np.uint8), iterations=1).astype(bool)
        src &= valid
        if int(np.count_nonzero(src)) < 20:
            continue
        matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        moved_img = cv2.warpAffine(base.lwir, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        moved_mask = (
            cv2.warpAffine(src.astype(np.uint8) * 255, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            > 0
        )
        paste_order.append((moved_img, moved_mask, src))
        erase_mask |= src
        pair_count += 1
        total_shift += abs(dx) + abs(dy)

    if pair_count == 0:
        return base.lwir.copy(), base.lwir_valid.copy(), {"component_snap_pairs": 0}

    bg = local_background_value(base.lwir, valid, erase_mask)
    out[erase_mask] = bg
    for moved_img, moved_mask, src in paste_order:
        out[moved_mask] = moved_img[moved_mask]
        out_valid[moved_mask] = True
        out_valid[src] = True
    return out, out_valid, {
        "component_snap_pairs": int(pair_count),
        "component_snap_total_shift_l1": float(total_shift),
        "component_snap_mode": mode,
    }


def edge_distance_stats(moving_edges: np.ndarray, fixed_edges: np.ndarray, roi: np.ndarray) -> tuple[float, float, float, int]:
    moving = moving_edges.astype(bool) & roi.astype(bool)
    fixed = fixed_edges.astype(bool) & roi.astype(bool)
    moving_count = int(np.count_nonzero(moving))
    fixed_count = int(np.count_nonzero(fixed))
    if moving_count < 20 or fixed_count < 20:
        return 12.0, 12.0, 12.0, moving_count
    distance = cv2.distanceTransform((~fixed).astype(np.uint8) * 255, cv2.DIST_L2, 3)
    values = distance[moving]
    return float(np.mean(values)), float(np.percentile(values, 70)), float(np.percentile(values, 90)), moving_count


def compactness(mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    area = int(np.count_nonzero(mask))
    if area == 0:
        return 0.0
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    largest = 0
    for idx in range(1, component_count):
        largest = max(largest, int(stats[idx, cv2.CC_STAT_AREA]))
    return float(largest / max(1, area))


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask.astype(bool))
    if len(xs) == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def centroid_distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float:
    if a is None or b is None:
        return 48.0
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def scene_class(raw_metrics: dict) -> str:
    support_pixels = int(raw_metrics.get("support_pixels", 0))
    thermal_pixels = int(raw_metrics.get("thermal_pixels", 0))
    rejected = int(raw_metrics.get("rejected_background_pixels", 0))
    depth_discontinuity = int(raw_metrics.get("depth_discontinuity_pixels", 0))
    ghost_ratio = float(raw_metrics.get("ghost_edge_ratio", 0.0))
    support_ratio = float(raw_metrics.get("support_ratio", 0.0))
    if rejected >= 3000 or thermal_pixels >= 9000:
        return "reflection_background"
    if support_pixels < 3500 or support_ratio < 0.012:
        return "weak_target_support"
    if depth_discontinuity >= 3500:
        return "depth_tearing"
    if ghost_ratio >= 0.90:
        return "double_edge_mismatch"
    return "general"


def support_args(args, candidate: CandidateOutput):
    local = copy.copy(args)
    local.max_support_area_frac = float(candidate.metadata.get("max_support_area_frac", args.max_support_area_frac))
    local.anti_ghost_erode_px = int(candidate.metadata.get("anti_ghost_erode_px", args.anti_ghost_erode_px))
    local.anti_ghost_feather_px = float(candidate.metadata.get("anti_ghost_feather_px", args.anti_ghost_feather_px))
    return local


def focus_context(item: dict, candidate: CandidateOutput, args) -> dict:
    valid = candidate.rgb_valid.astype(bool) & candidate.lwir_valid.astype(bool)
    rgb_offset = metadata_offset(candidate.metadata, "rgb_crop_offset_xy")
    target_size = (candidate.rgb.shape[1], candidate.rgb.shape[0])
    support = review_support_mask(candidate.rgb, valid, item.get("depth"), rgb_offset, item["row"].raw_rgb_anno_class_path)
    depth_crop, depth_valid = crop_depth_to_canvas(item, rgb_offset, target_size)
    mode = str(candidate.metadata.get("phase29_support_mode", "baseline"))

    local_args = support_args(args, candidate)
    anti_result = make_anti_ghost_support(candidate.rgb, candidate.lwir, valid, support, depth_crop, depth_valid, local_args)
    focus_mask = anti_result["mask"]
    thermal_mask = anti_result["thermal_mask"]
    rejected = anti_result["rejected_mask"]

    if mode == "reflection_guard":
        depth_mask = anti_result["depth_mask"]
        if np.any(depth_mask):
            depth_band = cv2.dilate(depth_mask.astype(np.uint8), np.ones((15, 15), np.uint8), iterations=1).astype(bool)
            guarded = focus_mask & depth_band
            if int(np.count_nonzero(guarded)) >= 40:
                rejected |= focus_mask & ~guarded
                focus_mask = guarded
    elif mode == "small_target_guard":
        if int(np.count_nonzero(focus_mask)) < 3000:
            thermal = review_thermal_target_mask(candidate.lwir, valid, support if np.any(support) else valid)
            compact = thermal & cv2.dilate((support | thermal).astype(np.uint8), np.ones((21, 21), np.uint8), iterations=1).astype(bool)
            if int(np.count_nonzero(compact)) >= 40:
                focus_mask = compact & valid
                thermal_mask = thermal
    elif mode == "edge_contamination_guard":
        if np.any(support):
            support_band = cv2.dilate(support.astype(np.uint8), np.ones((17, 17), np.uint8), iterations=1).astype(bool)
            guarded = focus_mask & support_band
            if int(np.count_nonzero(guarded)) >= 40:
                rejected |= focus_mask & ~guarded
                focus_mask = guarded
    elif mode == "depth_conservative":
        depth_mask = anti_result["depth_mask"]
        if int(np.count_nonzero(depth_mask)) >= 40:
            guarded = focus_mask & cv2.dilate(depth_mask.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1).astype(bool)
            if int(np.count_nonzero(guarded)) >= 40:
                rejected |= focus_mask & ~guarded
                focus_mask = guarded
    elif mode == "reflection_hard_reject":
        depth_mask = anti_result["depth_mask"].astype(bool)
        thermal = review_thermal_target_mask(candidate.lwir, valid, support if np.any(support) else valid)
        guard_base = focus_mask | thermal
        if np.any(depth_mask):
            guard_base &= cv2.dilate(depth_mask.astype(np.uint8), np.ones((19, 19), np.uint8), iterations=1).astype(bool)
        if np.any(support):
            guard_base &= cv2.dilate(support.astype(np.uint8), np.ones((25, 25), np.uint8), iterations=1).astype(bool)
        guard_base = cv2.morphologyEx(guard_base.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
        if int(np.count_nonzero(guard_base)) >= 40:
            rejected |= focus_mask & ~guard_base
            focus_mask = guard_base & valid
            thermal_mask = thermal
    elif mode == "reflection_tight_target":
        depth_mask = anti_result["depth_mask"].astype(bool)
        thermal = review_thermal_target_mask(candidate.lwir, valid, support if np.any(support) else valid)
        tight = thermal.copy()
        if np.any(support):
            tight &= cv2.dilate(support.astype(np.uint8), np.ones((21, 21), np.uint8), iterations=1).astype(bool)
        if np.any(depth_mask):
            tight &= cv2.dilate(depth_mask.astype(np.uint8), np.ones((13, 13), np.uint8), iterations=1).astype(bool)
        tight = cv2.morphologyEx(tight.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
        if int(np.count_nonzero(tight)) >= 40:
            rejected |= focus_mask & ~tight
            focus_mask = tight & valid
            thermal_mask = thermal
    elif mode == "small_target_conservative":
        depth_mask = anti_result["depth_mask"].astype(bool)
        thermal = review_thermal_target_mask(candidate.lwir, valid, support if np.any(support) else valid)
        support_band = cv2.dilate((support | thermal).astype(np.uint8), np.ones((17, 17), np.uint8), iterations=1).astype(bool)
        compact = thermal & support_band
        if np.any(depth_mask):
            compact &= cv2.dilate(depth_mask.astype(np.uint8), np.ones((11, 11), np.uint8), iterations=1).astype(bool)
        compact = cv2.erode(compact.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
        if int(np.count_nonzero(compact)) >= 25:
            rejected |= focus_mask & ~compact
            focus_mask = compact & valid
            thermal_mask = thermal
    elif mode == "foreground_tight":
        depth_mask = anti_result["depth_mask"].astype(bool)
        foreground = focus_mask.copy()
        if np.any(support):
            foreground &= cv2.dilate(support.astype(np.uint8), np.ones((19, 19), np.uint8), iterations=1).astype(bool)
        if np.any(thermal_mask):
            foreground |= thermal_mask & cv2.dilate(foreground.astype(np.uint8), np.ones((15, 15), np.uint8), iterations=1).astype(bool)
        if np.any(depth_mask):
            foreground &= cv2.dilate(depth_mask.astype(np.uint8), np.ones((15, 15), np.uint8), iterations=1).astype(bool)
        foreground = cv2.morphologyEx(foreground.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        if int(np.count_nonzero(foreground)) >= 40:
            rejected |= focus_mask & ~foreground
            focus_mask = foreground & valid
    elif mode == "depth_invalid_reject":
        depth_mask = anti_result["depth_mask"].astype(bool)
        valid_depth_band = cv2.dilate((depth_valid & valid).astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1).astype(bool)
        depth_guarded = focus_mask & valid_depth_band
        if np.any(depth_mask):
            depth_guarded &= cv2.dilate(depth_mask.astype(np.uint8), np.ones((15, 15), np.uint8), iterations=1).astype(bool)
        if int(np.count_nonzero(depth_guarded)) >= 40:
            rejected |= focus_mask & ~depth_guarded
            focus_mask = depth_guarded & valid
    elif mode == "v7_edge_local":
        depth_mask = anti_result["depth_mask"].astype(bool)
        thermal = review_thermal_target_mask(candidate.lwir, valid, support if np.any(support) else valid)
        local_core = focus_mask.copy()
        if np.any(support):
            local_core &= cv2.dilate(support.astype(np.uint8), np.ones((19, 19), np.uint8), iterations=1).astype(bool)
        if np.any(thermal):
            local_core |= thermal & cv2.dilate(local_core.astype(np.uint8), np.ones((13, 13), np.uint8), iterations=1).astype(bool)
        if np.any(depth_mask):
            local_core &= cv2.dilate(depth_mask.astype(np.uint8), np.ones((17, 17), np.uint8), iterations=1).astype(bool)
        local_core = cv2.morphologyEx(local_core.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        if int(np.count_nonzero(local_core)) >= 35:
            rejected |= focus_mask & ~local_core
            focus_mask = local_core & valid
            thermal_mask = thermal
    elif mode == "v7_reflection_core":
        depth_mask = anti_result["depth_mask"].astype(bool)
        thermal = review_thermal_target_mask(candidate.lwir, valid, support if np.any(support) else valid)
        local_core = thermal.copy()
        if np.any(support):
            local_core &= cv2.dilate(support.astype(np.uint8), np.ones((17, 17), np.uint8), iterations=1).astype(bool)
        if np.any(depth_mask):
            local_core &= cv2.dilate(depth_mask.astype(np.uint8), np.ones((11, 11), np.uint8), iterations=1).astype(bool)
        local_core &= cv2.dilate((depth_valid & valid).astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1).astype(bool)
        local_core = cv2.morphologyEx(local_core.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
        if int(np.count_nonzero(local_core)) >= 25:
            rejected |= focus_mask & ~local_core
            focus_mask = local_core & valid
            thermal_mask = thermal
    elif mode == "v7_weak_target_core":
        depth_mask = anti_result["depth_mask"].astype(bool)
        thermal = review_thermal_target_mask(candidate.lwir, valid, support if np.any(support) else valid)
        core = thermal.copy()
        if np.any(support):
            core |= support & cv2.dilate(thermal.astype(np.uint8), np.ones((15, 15), np.uint8), iterations=1).astype(bool)
        if np.any(depth_mask):
            core &= cv2.dilate(depth_mask.astype(np.uint8), np.ones((15, 15), np.uint8), iterations=1).astype(bool)
        core = cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
        if int(np.count_nonzero(core)) >= 25:
            rejected |= focus_mask & ~core
            focus_mask = core & valid
            thermal_mask = thermal

    if int(np.count_nonzero(focus_mask)) < 40:
        fallback = (support | thermal_mask) & valid
        if int(np.count_nonzero(fallback)) >= 40:
            focus_mask = fallback

    alpha = feather_alpha(focus_mask, valid, float(candidate.metadata.get("anti_ghost_feather_px", args.anti_ghost_feather_px)))
    risk_view, risk_metrics = tear_ghost_diagnostics(candidate.rgb, candidate.lwir, valid, focus_mask, alpha, depth_crop, depth_valid)
    return {
        "valid": valid,
        "support": support,
        "focus_mask": focus_mask.astype(bool),
        "alpha": alpha,
        "depth_crop": depth_crop,
        "depth_valid": depth_valid,
        "anti_result": anti_result,
        "thermal_mask": thermal_mask.astype(bool),
        "rejected_mask": rejected.astype(bool),
        "risk_view": risk_view,
        "risk_metrics": risk_metrics,
    }


def raw_selection_score(item: dict, candidate: CandidateOutput, context: dict, args) -> dict:
    valid = context["valid"]
    focus_mask = context["focus_mask"]
    depth_crop = context["depth_crop"]
    depth_valid = context["depth_valid"]
    risk_metrics = context["risk_metrics"]
    support_pixels = int(np.count_nonzero(focus_mask))
    support_ratio = float(support_pixels / max(1, focus_mask.size))
    roi = cv2.dilate(focus_mask.astype(np.uint8), np.ones((31, 31), np.uint8), iterations=1).astype(bool) & valid
    if support_pixels < 40:
        roi = valid

    edge_masks = contour_edges(candidate.rgb, candidate.lwir, valid, roi, depth_crop, depth_valid)
    lwir_to_rgb, lwir_edges = edge_distance(edge_masks["lwir_edges"], edge_masks["rgb_edges"], roi)
    rgb_to_lwir, rgb_edges = edge_distance(edge_masks["rgb_edges"], edge_masks["lwir_edges"], roi)
    lwir_to_depth, _lwir_edges_for_depth = edge_distance(edge_masks["lwir_edges"], edge_masks["depth_edges"], roi)
    depth_to_lwir, depth_edges = edge_distance(edge_masks["depth_edges"], edge_masks["lwir_edges"], roi)
    symmetric_edge_distance = 0.5 * (lwir_to_rgb + rgb_to_lwir)
    target_roi = cv2.dilate(focus_mask.astype(np.uint8), np.ones((17, 17), np.uint8), iterations=1).astype(bool) & valid
    if int(np.count_nonzero(target_roi)) < 80:
        target_roi = roi
    target_edges = contour_edges(candidate.rgb, candidate.lwir, valid, target_roi, depth_crop, depth_valid)
    target_lwir_to_rgb, target_lwir_p70, target_lwir_p90, target_lwir_edges = edge_distance_stats(
        target_edges["lwir_edges"], target_edges["rgb_edges"], target_roi
    )
    target_rgb_to_lwir, target_rgb_p70, target_rgb_p90, target_rgb_edges = edge_distance_stats(
        target_edges["rgb_edges"], target_edges["lwir_edges"], target_roi
    )
    target_lwir_to_depth, _target_lwir_depth_p70, target_lwir_depth_p90, _target_depth_lwir_edges = edge_distance_stats(
        target_edges["lwir_edges"], target_edges["depth_edges"], target_roi
    )
    target_depth_to_lwir, _target_depth_p70, target_depth_p90, target_depth_edges = edge_distance_stats(
        target_edges["depth_edges"], target_edges["lwir_edges"], target_roi
    )
    target_symmetric_edge_distance = 0.5 * (target_lwir_to_rgb + target_rgb_to_lwir)
    target_edge_p90 = 0.5 * (target_lwir_p90 + target_rgb_p90)
    edge_stability_score = abs(symmetric_edge_distance - target_symmetric_edge_distance) + max(0.0, target_edge_p90 - 6.0) * 0.25
    support_penalty = 0.0
    if support_ratio < float(args.min_support_ratio):
        support_penalty += (float(args.min_support_ratio) - support_ratio) * 260.0
    if support_ratio > float(args.max_raw_support_ratio):
        support_penalty += (support_ratio - float(args.max_raw_support_ratio)) * 180.0
    support_compactness = compactness(focus_mask)
    compactness_penalty = max(0.0, 0.55 - support_compactness) * 4.0
    rejected_background = int(np.count_nonzero(context["rejected_mask"]))
    thermal_pixels = int(np.count_nonzero(context["thermal_mask"]))
    tear_pixels = int(risk_metrics.get("tear_risk_pixels", 0))
    ghost_ratio = float(risk_metrics.get("ghost_edge_ratio", 1.0))
    alpha_overlap = float(risk_metrics.get("alpha_edge_overlap", 1.0))
    depth_discontinuity = int(risk_metrics.get("depth_discontinuity_pixels", 0))
    fill_penalty = 0.22 if bool(candidate.metadata.get("fill_depth_border", False)) and depth_discontinuity > 3000 else 0.0
    support_core = context["support"].astype(bool) & valid
    thermal_core = context["thermal_mask"].astype(bool) & valid
    depth_core = context["anti_result"].get("depth_mask", np.zeros_like(valid)).astype(bool) & valid
    thermal_count = int(np.count_nonzero(thermal_core))
    support_count = int(np.count_nonzero(support_core))
    depth_count = int(np.count_nonzero(depth_core))
    if thermal_count > 0:
        thermal_band = cv2.dilate(thermal_core.astype(np.uint8), np.ones((17, 17), np.uint8), iterations=1).astype(bool)
        support_band = cv2.dilate(support_core.astype(np.uint8), np.ones((17, 17), np.uint8), iterations=1).astype(bool)
        depth_band = cv2.dilate(depth_core.astype(np.uint8), np.ones((17, 17), np.uint8), iterations=1).astype(bool)
        thermal_support_overlap = float(np.count_nonzero(thermal_core & support_band) / max(1, thermal_count))
        thermal_depth_overlap = float(np.count_nonzero(thermal_core & depth_band) / max(1, thermal_count)) if depth_count > 0 else 0.0
        support_thermal_overlap = float(np.count_nonzero(support_core & thermal_band) / max(1, support_count)) if support_count > 0 else 0.0
    else:
        thermal_support_overlap = 0.0
        thermal_depth_overlap = 0.0
        support_thermal_overlap = 0.0
    thermal_centroid = centroid(thermal_core)
    support_centroid = centroid((support_core | depth_core) & valid)
    thermal_support_centroid_distance = centroid_distance(thermal_centroid, support_centroid)
    depth_edge_penalty = 0.0
    if depth_edges >= 20:
        depth_edge_penalty = 0.18 * min(12.0, lwir_to_depth) + 0.08 * min(12.0, depth_to_lwir)
    overlap_penalty = max(0.0, 0.55 - thermal_support_overlap) * 1.15 + max(0.0, 0.35 - thermal_depth_overlap) * 0.55
    centroid_penalty = min(48.0, thermal_support_centroid_distance) / 52.0
    score = (
        symmetric_edge_distance
        + ghost_ratio * 2.3
        + alpha_overlap * 1.2
        + tear_pixels / 1600.0
        + depth_discontinuity / 2800.0
        + rejected_background / 6500.0
        + thermal_pixels / 26000.0
        + depth_edge_penalty
        + overlap_penalty
        + centroid_penalty
        + support_penalty
        + compactness_penalty
        + fill_penalty
    )
    return {
        "raw_score": float(score),
        "raw_symmetric_edge_distance": float(symmetric_edge_distance),
        "raw_lwir_to_rgb_edge_distance": float(lwir_to_rgb),
        "raw_rgb_to_lwir_edge_distance": float(rgb_to_lwir),
        "raw_lwir_to_depth_edge_distance": float(lwir_to_depth),
        "raw_depth_to_lwir_edge_distance": float(depth_to_lwir),
        "target_local_symmetric_edge_distance": float(target_symmetric_edge_distance),
        "target_local_lwir_to_rgb_edge_distance": float(target_lwir_to_rgb),
        "target_local_rgb_to_lwir_edge_distance": float(target_rgb_to_lwir),
        "target_local_lwir_to_depth_edge_distance": float(target_lwir_to_depth),
        "target_local_depth_to_lwir_edge_distance": float(target_depth_to_lwir),
        "target_local_edge_p70": float(0.5 * (target_lwir_p70 + target_rgb_p70)),
        "target_local_edge_p90": float(target_edge_p90),
        "target_local_lwir_edge_pixels": int(target_lwir_edges),
        "target_local_rgb_edge_pixels": int(target_rgb_edges),
        "target_local_depth_edge_pixels": int(target_depth_edges),
        "edge_stability_score": float(edge_stability_score),
        "target_local_depth_edge_p90": float(0.5 * (target_lwir_depth_p90 + target_depth_p90)),
        "raw_lwir_edge_pixels": int(lwir_edges),
        "raw_rgb_edge_pixels": int(rgb_edges),
        "raw_depth_edge_pixels": int(depth_edges),
        "support_pixels": support_pixels,
        "support_ratio": support_ratio,
        "support_compactness": support_compactness,
        "thermal_support_overlap": thermal_support_overlap,
        "thermal_depth_overlap": thermal_depth_overlap,
        "support_thermal_overlap": support_thermal_overlap,
        "thermal_support_centroid_distance_px": thermal_support_centroid_distance,
        "thermal_pixels": thermal_pixels,
        "rejected_background_pixels": rejected_background,
        "tear_risk_pixels": tear_pixels,
        "ghost_edge_ratio": ghost_ratio,
        "alpha_edge_overlap": alpha_overlap,
        "depth_discontinuity_pixels": depth_discontinuity,
        "support_penalty": float(support_penalty),
        "compactness_penalty": float(compactness_penalty),
        "fill_penalty": float(fill_penalty),
        "depth_edge_penalty": float(depth_edge_penalty),
        "overlap_penalty": float(overlap_penalty),
        "centroid_penalty": float(centroid_penalty),
    }


def contour_edges(
    rgb_bgr: np.ndarray,
    lwir_u8: np.ndarray,
    valid: np.ndarray,
    roi: np.ndarray,
    depth_crop: np.ndarray,
    depth_valid: np.ndarray,
) -> dict:
    from run_darklight import auto_edges, enhance_lowlight_bgr
    from run_phase28 import depth_discontinuity_edges

    enhanced = enhance_lowlight_bgr(rgb_bgr)
    rgb_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    rgb_edges = (auto_edges(rgb_gray) > 0) & roi.astype(bool)
    lwir_edges = (auto_edges(lwir_u8) > 0) & roi.astype(bool)
    depth_edges = depth_discontinuity_edges(depth_crop, depth_valid, valid.astype(bool)) & roi.astype(bool)
    return {"rgb_edges": rgb_edges, "lwir_edges": lwir_edges, "depth_edges": depth_edges}


def selection_family(record: dict) -> str:
    name = str(record["candidate"].name)
    fill = bool(record["candidate"].metadata.get("fill_depth_border", False))
    if name == "p29_baseline":
        return "baseline"
    if name.startswith("p29_raw_shift_") and fill:
        return "raw_shift_fill"
    if name.startswith("p29_raw_shift_"):
        return "raw_shift_nofill"
    if name.startswith("p29_v6_risk_shift_") and fill:
        return "v6_risk_shift_fill"
    if name.startswith("p29_v7_edge_shift_") and fill:
        return "v7_edge_shift_fill"
    if name.startswith("p29_v7_"):
        return "v7_guard"
    if name.startswith("p29_v8_component_"):
        return "v8_component_snap"
    if name == "p29_v8_depth_projection_only":
        return "v8_depth_projection"
    if name.startswith("p29_v9_"):
        return "v9_support_gated"
    if name == "p29_small_target_guard":
        return "small_target_guard"
    return "diagnostic_guard"


def safety_rejection(record: dict, baseline: dict, args) -> str:
    family = selection_family(record)
    if family in {"baseline"}:
        return ""
    if family == "raw_shift_nofill":
        return "nofill shift candidates are diagnostic-only by default"
    if family == "diagnostic_guard" and not bool(args.allow_diagnostic_guard_selection):
        return "depth/reflection/edge guard candidates are diagnostic-only by default"

    raw = record["raw_metrics"]
    base = baseline["raw_metrics"]
    baseline_score = float(base["raw_score"])
    score = float(raw["raw_score"])
    score_margin = float(args.guard_score_margin) if family == "small_target_guard" else float(args.raw_score_margin)
    if (
        family == "raw_shift_fill"
        and int(base["support_pixels"]) <= int(args.weak_support_pixels)
        and baseline_score - score >= float(args.weak_support_raw_score_margin)
    ):
        score_margin = float(args.weak_support_raw_score_margin)
    if baseline_score < float(args.low_risk_raw_score):
        return f"baseline raw score {baseline_score:.3f} is below low-risk threshold"
    if baseline_score - score < score_margin:
        return f"raw improvement {baseline_score - score:.4f} is below {score_margin:.4f}"

    baseline_edge = float(base["raw_symmetric_edge_distance"])
    candidate_edge = float(raw["raw_symmetric_edge_distance"])
    if candidate_edge > baseline_edge - float(args.raw_edge_margin):
        return f"raw symmetric edge improvement {baseline_edge - candidate_edge:.4f} is below {float(args.raw_edge_margin):.4f}"

    baseline_lwir_edges = max(1, int(base["raw_lwir_edge_pixels"]))
    candidate_lwir_edges = int(raw["raw_lwir_edge_pixels"])
    edge_growth = candidate_lwir_edges / baseline_lwir_edges
    edge_growth_limit = float(args.max_lwir_edge_growth)
    if family == "small_target_guard" and float(base.get("thermal_support_overlap", 1.0)) >= float(args.reflection_overlap_threshold):
        edge_growth_limit = float(args.max_guard_lwir_edge_growth)
    if edge_growth > edge_growth_limit:
        return f"LWIR edge growth {edge_growth:.3f} exceeds {edge_growth_limit:.3f}"
    if edge_growth < float(args.min_lwir_edge_retain):
        return f"LWIR edge retain ratio {edge_growth:.3f} is below {float(args.min_lwir_edge_retain):.3f}"

    baseline_support = max(1, int(base["support_pixels"]))
    candidate_support = int(raw["support_pixels"])
    support_ratio = candidate_support / baseline_support
    if support_ratio < float(args.min_selection_support_retain):
        return f"support retain ratio {support_ratio:.3f} is below {float(args.min_selection_support_retain):.3f}"
    if support_ratio > float(args.max_selection_support_growth):
        return f"support growth ratio {support_ratio:.3f} exceeds {float(args.max_selection_support_growth):.3f}"

    if family == "small_target_guard" and baseline_support < int(args.min_guard_baseline_support_pixels):
        return f"small-target guard blocked because baseline support {baseline_support} is too small"
    return ""


def selection_adjusted_score(record: dict, baseline: dict, args) -> float:
    raw = record["raw_metrics"]
    score = float(raw["raw_score"])
    baseline_dx = float(baseline["candidate"].metadata.get("depth_registration_dx_px", 0.0))
    baseline_dy = float(baseline["candidate"].metadata.get("depth_registration_dy_px", 0.0))
    dx = float(record["candidate"].metadata.get("depth_registration_dx_px", baseline_dx))
    dy = float(record["candidate"].metadata.get("depth_registration_dy_px", baseline_dy))
    shift_penalty = (abs(dx - baseline_dx) + abs(dy - baseline_dy)) * float(args.shift_prior_weight)
    family = selection_family(record)
    if family == "small_target_guard":
        shift_penalty += float(args.guard_prior_penalty)
    return score + shift_penalty


def shift_delta_from_baseline(record: dict, baseline: dict) -> tuple[float, float, float]:
    baseline_dx = float(baseline["candidate"].metadata.get("depth_registration_dx_px", 0.0))
    baseline_dy = float(baseline["candidate"].metadata.get("depth_registration_dy_px", 0.0))
    dx = float(record["candidate"].metadata.get("depth_registration_dx_px", baseline_dx))
    dy = float(record["candidate"].metadata.get("depth_registration_dy_px", baseline_dy))
    delta_dx = dx - baseline_dx
    delta_dy = dy - baseline_dy
    return delta_dx, delta_dy, abs(delta_dx) + abs(delta_dy)


def selector_debug_row(record: dict, baseline: dict, eligible: bool, reason: str, adjusted_score: float, rule: str) -> dict:
    raw = record["raw_metrics"]
    base = baseline["raw_metrics"]
    delta_dx, delta_dy, shift_l1 = shift_delta_from_baseline(record, baseline)
    base_lwir_edges = max(1, int(base.get("raw_lwir_edge_pixels", 1)))
    support_base = max(1, int(base.get("support_pixels", 1)))
    return {
        "phase29_selector_eligible": bool(eligible),
        "phase29_selector_rejection": reason,
        "phase29_selector_adjusted_score": float(adjusted_score),
        "phase29_selector_rule": rule,
        "phase29_selector_shift_delta_dx": float(delta_dx),
        "phase29_selector_shift_delta_dy": float(delta_dy),
        "phase29_selector_shift_l1": float(shift_l1),
        "phase29_selector_raw_improvement": float(base.get("raw_score", 0.0) - raw.get("raw_score", 0.0)),
        "phase29_selector_edge_growth": float(int(raw.get("raw_lwir_edge_pixels", 0)) / base_lwir_edges),
        "phase29_selector_support_ratio": float(int(raw.get("support_pixels", 0)) / support_base),
        "phase29_selector_target_edge_gain": float(
            base.get("target_local_symmetric_edge_distance", 12.0) - raw.get("target_local_symmetric_edge_distance", 12.0)
        ),
        "phase29_selector_target_p90_gain": float(base.get("target_local_edge_p90", 12.0) - raw.get("target_local_edge_p90", 12.0)),
        "phase29_selector_context_edge_delta": float(
            raw.get("raw_symmetric_edge_distance", 12.0) - base.get("raw_symmetric_edge_distance", 12.0)
        ),
    }


def safety_rejection_v2(record: dict, baseline: dict, args) -> tuple[str, str]:
    family = selection_family(record)
    if family == "baseline":
        return "", "baseline"
    if family == "raw_shift_nofill":
        return "nofill shift candidates are diagnostic-only by default", "nofill_block"
    if family == "diagnostic_guard" and not bool(args.allow_diagnostic_guard_selection):
        return "depth/reflection/edge guard candidates are diagnostic-only by default", "diagnostic_guard_block"
    if family == "v6_risk_shift_fill":
        return "v6 risk-shift candidates require the selector v4 wide-risk gate", "v6_requires_v4_gate"
    if family in {"v7_edge_shift_fill", "v7_guard"}:
        return "v7 edge candidates require the selector v5 target-local gate", "v7_requires_v5_gate"
    raw = record["raw_metrics"]
    base = baseline["raw_metrics"]
    scene = scene_class(base)
    _delta_dx, _delta_dy, shift_l1 = shift_delta_from_baseline(record, baseline)

    if family == "small_target_guard":
        if scene == "depth_tearing":
            return "small-target guard is blocked in depth-tearing scenes so local depth correction can be tested", "small_target_depth_block"
        reason = safety_rejection(record, baseline, args)
        return reason, "small_target_guard" if not reason else "small_target_guard_block"
    if family != "raw_shift_fill":
        reason = safety_rejection(record, baseline, args)
        return reason, "generic_guard" if not reason else "generic_block"

    baseline_score = float(base["raw_score"])
    score = float(raw["raw_score"])
    improvement = baseline_score - score
    baseline_edge = float(base["raw_symmetric_edge_distance"])
    candidate_edge = float(raw["raw_symmetric_edge_distance"])
    baseline_lwir_edges = max(1, int(base["raw_lwir_edge_pixels"]))
    candidate_lwir_edges = int(raw["raw_lwir_edge_pixels"])
    edge_growth = candidate_lwir_edges / baseline_lwir_edges
    baseline_support = max(1, int(base["support_pixels"]))
    support_ratio = int(raw["support_pixels"]) / baseline_support
    delta_dx, delta_dy, shift_l1 = shift_delta_from_baseline(record, baseline)
    centroid_delta = abs(
        float(raw.get("thermal_support_centroid_distance_px", 0.0))
        - float(base.get("thermal_support_centroid_distance_px", 0.0))
    )

    if support_ratio < float(args.min_selection_support_retain):
        return f"support retain ratio {support_ratio:.3f} is below {float(args.min_selection_support_retain):.3f}", "support_block"
    if support_ratio > float(args.max_selection_support_growth):
        return f"support growth ratio {support_ratio:.3f} exceeds {float(args.max_selection_support_growth):.3f}", "support_block"
    if edge_growth < float(args.min_lwir_edge_retain):
        return f"LWIR edge retain ratio {edge_growth:.3f} is below {float(args.min_lwir_edge_retain):.3f}", "edge_growth_block"

    if scene == "depth_tearing" and float(args.depth_tearing_raw_score_min) <= baseline_score <= float(args.depth_tearing_raw_score_max):
        if improvement < float(args.depth_tearing_raw_margin):
            return f"depth-tearing raw improvement {improvement:.4f} is below {float(args.depth_tearing_raw_margin):.4f}", "depth_tearing_block"
        if candidate_edge > baseline_edge + float(args.depth_tearing_raw_edge_slack):
            return (
                f"depth-tearing raw edge slack {candidate_edge - baseline_edge:.4f} exceeds {float(args.depth_tearing_raw_edge_slack):.4f}",
                "depth_tearing_block",
            )
        if edge_growth > float(args.depth_tearing_max_lwir_edge_growth):
            return f"depth-tearing LWIR edge growth {edge_growth:.3f} exceeds {float(args.depth_tearing_max_lwir_edge_growth):.3f}", "depth_tearing_block"
        if abs(delta_dx) > float(args.depth_tearing_max_abs_shift) or abs(delta_dy) > float(args.depth_tearing_max_abs_shift):
            return "depth-tearing candidate shift exceeds local safe range", "depth_tearing_block"
        return "", "depth_tearing_v2"

    if (
        scene == "double_edge_mismatch"
        and baseline_score >= float(args.double_edge_raw_score_min)
        and float(base.get("ghost_edge_ratio", 0.0)) >= float(args.double_edge_ghost_ratio_min)
    ):
        if str(args.candidate_grid) != "wide-safe" and str(args.run_mode) != "ceiling-study":
            return "severe double-edge recovery requires wide-safe candidate grid", "double_edge_block"
        if shift_l1 < float(args.double_edge_min_shift_l1) or shift_l1 > float(args.double_edge_max_shift_l1):
            return f"double-edge shift l1 {shift_l1:.3f} outside wide-safe range", "double_edge_block"
        if improvement < float(args.double_edge_raw_margin):
            return f"double-edge raw improvement {improvement:.4f} is below {float(args.double_edge_raw_margin):.4f}", "double_edge_block"
        if edge_growth > float(args.double_edge_max_lwir_edge_growth):
            return f"double-edge LWIR edge growth {edge_growth:.3f} exceeds {float(args.double_edge_max_lwir_edge_growth):.3f}", "double_edge_block"
        if centroid_delta > float(args.double_edge_max_centroid_delta_px):
            return f"double-edge centroid delta {centroid_delta:.3f}px exceeds {float(args.double_edge_max_centroid_delta_px):.3f}px", "double_edge_block"
        return "", "severe_double_edge_v2"

    if shift_l1 > float(args.generic_max_shift_l1):
        return f"generic raw-shift l1 {shift_l1:.3f} exceeds {float(args.generic_max_shift_l1):.3f}", "generic_wide_block"

    reason = safety_rejection(record, baseline, args)
    return reason, "generic_v2" if not reason else "generic_block"


def selection_adjusted_score_v2(record: dict, baseline: dict, args, rule: str) -> float:
    raw = record["raw_metrics"]
    score = selection_adjusted_score(record, baseline, args)
    if rule == "depth_tearing_v2":
        return score + float(raw.get("raw_lwir_to_depth_edge_distance", 12.0)) * float(args.depth_tearing_depth_edge_weight)
    if rule == "severe_double_edge_v2":
        _, _delta_dy, shift_l1 = shift_delta_from_baseline(record, baseline)
        edge_growth = int(raw.get("raw_lwir_edge_pixels", 0)) / max(1, int(baseline["raw_metrics"].get("raw_lwir_edge_pixels", 1)))
        return score + edge_growth * float(args.double_edge_edge_growth_weight) + shift_l1 * float(args.double_edge_shift_weight)
    return score


def safety_rejection_v4(record: dict, baseline: dict, args) -> tuple[str, str]:
    reason, rule = safety_rejection_v2(record, baseline, args)
    if not reason:
        return "", rule

    family = selection_family(record)
    if family != "v6_risk_shift_fill":
        return reason, rule

    raw = record["raw_metrics"]
    base = baseline["raw_metrics"]
    scene = scene_class(base)
    delta_dx, delta_dy, shift_l1 = shift_delta_from_baseline(record, baseline)
    baseline_score = float(base["raw_score"])
    score = float(raw["raw_score"])
    improvement = baseline_score - score
    baseline_edge = float(base["raw_symmetric_edge_distance"])
    candidate_edge = float(raw["raw_symmetric_edge_distance"])
    raw_edge_gain = baseline_edge - candidate_edge
    baseline_lwir_edges = max(1, int(base["raw_lwir_edge_pixels"]))
    edge_growth = int(raw["raw_lwir_edge_pixels"]) / baseline_lwir_edges
    baseline_support = max(1, int(base["support_pixels"]))
    support_ratio = int(raw["support_pixels"]) / baseline_support
    ghost_delta = float(base.get("ghost_edge_ratio", 0.0)) - float(raw.get("ghost_edge_ratio", 0.0))
    tear_delta = int(base.get("tear_risk_pixels", 0)) - int(raw.get("tear_risk_pixels", 0))
    if scene == "reflection_background":
        return "v6 wide-risk shift is blocked in reflection/background scenes; use target/support guards only", "v6_reflection_block"
    if scene == "depth_tearing":
        return "v6 wide-risk shift is blocked in depth-tearing scenes; v2 depth-tearing gate is the only allowed shift path", "v6_depth_tearing_block"
    edge_contamination_like = (
        scene == "general"
        and int(base.get("thermal_pixels", 0)) >= int(args.v6_edge_contamination_min_thermal_pixels)
        and float(base.get("support_compactness", 1.0)) <= float(args.v6_edge_contamination_max_compactness)
        and float(base.get("ghost_edge_ratio", 0.0)) <= float(args.v6_edge_contamination_max_ghost_ratio)
        and baseline_score <= float(args.v6_edge_contamination_max_raw_score)
    )
    if scene == "general" and not edge_contamination_like:
        return "v6 wide-risk shift is blocked in general scenes because raw background edges can over-select large shifts", "v6_general_block"
    if delta_dx < float(args.v6_min_positive_delta_dx):
        return f"v6 risk shift dx delta {delta_dx:.3f} is below {float(args.v6_min_positive_delta_dx):.3f}", "v6_direction_block"

    high_risk_scene = scene in {"double_edge_mismatch", "weak_target_support"}

    raw_margin = float(args.v6_extended_raw_margin) if high_risk_scene else float(args.v6_low_risk_raw_margin)
    if edge_contamination_like:
        raw_margin = float(args.v6_edge_contamination_raw_margin)
    if improvement < raw_margin:
        return f"v6 raw improvement {improvement:.4f} is below {raw_margin:.4f}", "v6_raw_margin_block"

    allowed_shift_l1 = float(args.v6_generic_max_shift_l1)
    if high_risk_scene:
        allowed_shift_l1 = float(args.v6_high_risk_max_shift_l1)
    if scene == "weak_target_support":
        allowed_shift_l1 = min(allowed_shift_l1, float(args.v6_weak_support_max_shift_l1))
    if shift_l1 > allowed_shift_l1:
        return f"v6 shift l1 {shift_l1:.3f} exceeds {allowed_shift_l1:.3f}", "v6_shift_block"
    if abs(delta_dx) > float(args.v6_max_abs_shift) or abs(delta_dy) > float(args.v6_max_abs_shift):
        return "v6 candidate absolute shift exceeds safe range", "v6_shift_block"

    if support_ratio < float(args.v6_min_support_retain):
        return f"v6 support retain ratio {support_ratio:.3f} is below {float(args.v6_min_support_retain):.3f}", "v6_support_block"
    if support_ratio > float(args.v6_max_support_growth):
        return f"v6 support growth ratio {support_ratio:.3f} exceeds {float(args.v6_max_support_growth):.3f}", "v6_support_block"
    if edge_growth < float(args.min_lwir_edge_retain):
        return f"v6 LWIR edge retain ratio {edge_growth:.3f} is below {float(args.min_lwir_edge_retain):.3f}", "v6_edge_growth_block"
    if edge_growth > float(args.v6_max_lwir_edge_growth):
        return f"v6 LWIR edge growth {edge_growth:.3f} exceeds {float(args.v6_max_lwir_edge_growth):.3f}", "v6_edge_growth_block"
    if candidate_edge > baseline_edge + float(args.v6_raw_edge_slack):
        return f"v6 raw edge slack {candidate_edge - baseline_edge:.4f} exceeds {float(args.v6_raw_edge_slack):.4f}", "v6_raw_edge_block"

    has_visual_gain = raw_edge_gain >= float(args.v6_extended_raw_edge_margin)
    has_visual_gain = has_visual_gain or ghost_delta >= float(args.v6_min_ghost_reduction)
    has_visual_gain = has_visual_gain or tear_delta >= int(args.v6_min_tear_reduction_pixels)
    if not has_visual_gain:
        return "v6 candidate lacks raw-edge, ghost, or tear-risk gain", "v6_visual_gain_block"
    if ghost_delta < -float(args.v6_max_ghost_increase):
        return f"v6 ghost ratio increases by {-ghost_delta:.4f}", "v6_ghost_block"

    if family == "v6_risk_shift_fill":
        return "", "v6_risk_shift"
    return "", "v6_extended_raw_shift"


def selection_adjusted_score_v4(record: dict, baseline: dict, args, rule: str) -> float:
    score = selection_adjusted_score_v2(record, baseline, args, rule)
    if rule in {"v6_risk_shift", "v6_extended_raw_shift"}:
        raw = record["raw_metrics"]
        base = baseline["raw_metrics"]
        _delta_dx, _delta_dy, shift_l1 = shift_delta_from_baseline(record, baseline)
        edge_growth = int(raw.get("raw_lwir_edge_pixels", 0)) / max(1, int(base.get("raw_lwir_edge_pixels", 1)))
        ghost_penalty = max(0.0, float(raw.get("ghost_edge_ratio", 0.0)) - float(base.get("ghost_edge_ratio", 0.0)))
        return (
            float(raw["raw_score"])
            + shift_l1 * float(args.v6_shift_weight)
            + edge_growth * float(args.v6_edge_growth_weight)
            + ghost_penalty * float(args.v6_ghost_increase_weight)
        )
    return score


def deghost_acceptance_fusion(
    rgb_bgr: np.ndarray,
    lwir_u8: np.ndarray,
    valid: np.ndarray,
    focus_mask: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    base = enhance_lowlight_bgr(rgb_bgr).astype(np.float32)
    valid = valid.astype(bool)
    focus = focus_mask.astype(bool) & valid
    heat = cv2.applyColorMap(normalize_u8(lwir_u8), cv2.COLORMAP_TURBO).astype(np.float32)
    alpha_map = np.clip(alpha, 0.0, 1.0)[:, :, None] * 0.075
    out = base * (1.0 - alpha_map) + heat * alpha_map
    if np.any(focus):
        edge_support = cv2.dilate(focus.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1).astype(bool) & valid
        lwir_edges = (auto_edges(lwir_u8) > 0) & edge_support
        lwir_edges = cv2.morphologyEx(lwir_edges.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
        lwir_edges = cv2.dilate(lwir_edges.astype(np.uint8), np.ones((2, 2), np.uint8), iterations=1).astype(bool)
        out_u8 = np.clip(out, 0, 255).astype(np.uint8)
        edge_color = np.array([0, 220, 255], dtype=np.uint8)
        out_u8[lwir_edges] = ((out_u8[lwir_edges].astype(np.float32) * 0.35) + edge_color.astype(np.float32) * 0.65).astype(np.uint8)
        return out_u8
    return np.clip(out, 0, 255).astype(np.uint8)


def safety_rejection_v5(record: dict, baseline: dict, args) -> tuple[str, str]:
    reason, rule = safety_rejection_v4(record, baseline, args)
    if not reason:
        return "", rule

    family = selection_family(record)
    raw = record["raw_metrics"]
    base = baseline["raw_metrics"]
    scene = scene_class(base)

    if family not in {"v7_edge_shift_fill", "v7_guard"}:
        return reason, rule

    baseline_score = float(base["raw_score"])
    score = float(raw["raw_score"])
    score_delta = score - baseline_score
    target_gain = float(base.get("target_local_symmetric_edge_distance", 12.0)) - float(
        raw.get("target_local_symmetric_edge_distance", 12.0)
    )
    target_p90_gain = float(base.get("target_local_edge_p90", 12.0)) - float(raw.get("target_local_edge_p90", 12.0))
    context_edge_delta = float(raw.get("raw_symmetric_edge_distance", 12.0)) - float(
        base.get("raw_symmetric_edge_distance", 12.0)
    )
    ghost_delta = float(base.get("ghost_edge_ratio", 0.0)) - float(raw.get("ghost_edge_ratio", 0.0))
    tear_delta = int(base.get("tear_risk_pixels", 0)) - int(raw.get("tear_risk_pixels", 0))
    baseline_lwir_edges = max(1, int(base.get("raw_lwir_edge_pixels", 1)))
    edge_growth = int(raw.get("raw_lwir_edge_pixels", 0)) / baseline_lwir_edges
    baseline_support = max(1, int(base.get("support_pixels", 1)))
    support_ratio = int(raw.get("support_pixels", 0)) / baseline_support
    delta_dx, delta_dy, shift_l1 = shift_delta_from_baseline(record, baseline)

    if score_delta > float(args.v7_raw_score_slack):
        return f"v7 raw score worsens by {score_delta:.4f}, above slack {float(args.v7_raw_score_slack):.4f}", "v7_raw_score_block"
    if context_edge_delta > float(args.v7_context_edge_slack):
        return f"v7 context edge worsens by {context_edge_delta:.4f}, above slack {float(args.v7_context_edge_slack):.4f}", "v7_context_edge_block"
    if support_ratio < float(args.v7_min_support_retain):
        return f"v7 support retain ratio {support_ratio:.3f} is below {float(args.v7_min_support_retain):.3f}", "v7_support_block"
    if support_ratio > float(args.v7_max_support_growth):
        return f"v7 support growth ratio {support_ratio:.3f} exceeds {float(args.v7_max_support_growth):.3f}", "v7_support_block"
    if edge_growth < float(args.min_lwir_edge_retain):
        return f"v7 LWIR edge retain ratio {edge_growth:.3f} is below {float(args.min_lwir_edge_retain):.3f}", "v7_edge_growth_block"
    if edge_growth > float(args.v7_max_lwir_edge_growth):
        return f"v7 LWIR edge growth {edge_growth:.3f} exceeds {float(args.v7_max_lwir_edge_growth):.3f}", "v7_edge_growth_block"
    if ghost_delta < -float(args.v7_max_ghost_increase):
        return f"v7 ghost ratio increases by {-ghost_delta:.4f}", "v7_ghost_block"

    if family == "v7_edge_shift_fill":
        if shift_l1 > float(args.v7_edge_max_shift_l1):
            return f"v7 edge shift l1 {shift_l1:.3f} exceeds {float(args.v7_edge_max_shift_l1):.3f}", "v7_shift_block"
        if abs(delta_dx) > float(args.v7_max_abs_shift) or abs(delta_dy) > float(args.v7_max_abs_shift):
            return "v7 edge shift absolute delta exceeds safe range", "v7_shift_block"
        target_margin = float(args.v7_target_local_edge_margin)
        if scene == "reflection_background":
            target_margin = float(args.v7_reflection_target_edge_margin)
        if target_gain < target_margin:
            return f"v7 target-local edge gain {target_gain:.4f} is below {target_margin:.4f}", "v7_target_edge_block"
        if target_p90_gain < -float(args.v7_target_p90_slack):
            return f"v7 target-local p90 worsens by {-target_p90_gain:.4f}", "v7_target_p90_block"
        if scene == "general" and float(base.get("support_compactness", 1.0)) > float(args.v7_general_max_compactness):
            return "v7 general-scene shift blocked because support is already compact enough", "v7_general_stability_block"
        if scene == "general" and float(raw.get("target_local_symmetric_edge_distance", 12.0)) > float(args.v7_general_max_target_edge):
            return "v7 general-scene shift blocked because target-local edge remains too large", "v7_general_target_block"
        if scene == "general" and float(raw.get("target_local_edge_p90", 12.0)) > float(args.v7_general_max_target_p90):
            return "v7 general-scene shift blocked because target-local p90 remains too large", "v7_general_p90_block"
        return "", "v7_target_local_shift"

    if scene == "general":
        return "v7 support guard is blocked in general scenes; use edge-shift evidence instead", "v7_guard_general_block"
    if target_gain < float(args.v7_guard_target_edge_margin) and ghost_delta < float(args.v7_min_ghost_reduction) and tear_delta < int(args.v7_min_tear_reduction_pixels):
        return "v7 guard lacks target-edge, ghost, or tear improvement", "v7_guard_gain_block"
    return "", "v7_support_guard"


def selection_adjusted_score_v5(record: dict, baseline: dict, args, rule: str) -> float:
    score = selection_adjusted_score_v4(record, baseline, args, rule)
    if rule not in {"v7_target_local_shift", "v7_support_guard"}:
        return score
    raw = record["raw_metrics"]
    base = baseline["raw_metrics"]
    _delta_dx, _delta_dy, shift_l1 = shift_delta_from_baseline(record, baseline)
    edge_growth = int(raw.get("raw_lwir_edge_pixels", 0)) / max(1, int(base.get("raw_lwir_edge_pixels", 1)))
    ghost_penalty = max(0.0, float(raw.get("ghost_edge_ratio", 0.0)) - float(base.get("ghost_edge_ratio", 0.0)))
    target_gain = float(base.get("target_local_symmetric_edge_distance", 12.0)) - float(
        raw.get("target_local_symmetric_edge_distance", 12.0)
    )
    target_p90_gain = float(base.get("target_local_edge_p90", 12.0)) - float(raw.get("target_local_edge_p90", 12.0))
    return (
        float(raw.get("raw_score", 12.0))
        - max(0.0, target_gain) * float(args.v7_target_edge_weight)
        - max(0.0, target_p90_gain) * float(args.v7_target_p90_weight)
        + float(raw.get("edge_stability_score", 0.0)) * float(args.v7_stability_weight)
        + shift_l1 * float(args.v7_shift_weight)
        + edge_growth * float(args.v7_edge_growth_weight)
        + ghost_penalty * float(args.v7_ghost_increase_weight)
    )


def safety_rejection_v6(record: dict, baseline: dict, args) -> tuple[str, str]:
    family = selection_family(record)
    if family != "v9_support_gated":
        return safety_rejection_v5(record, baseline, args)

    raw = record["raw_metrics"]
    base = baseline["raw_metrics"]
    mode = str(record["candidate"].metadata.get("phase29_v9_gate_mode", record["candidate"].metadata.get("phase29_support_mode", "")))
    scene = scene_class(base)
    gate_pixels = int(record["candidate"].metadata.get("phase29_v9_gate_pixels", 0) or 0)
    raw_lwir_edges = int(raw.get("raw_lwir_edge_pixels", 0))
    base_thermal = int(base.get("thermal_pixels", 0))
    base_rejected = int(base.get("rejected_background_pixels", 0))
    base_score = float(base.get("raw_score", 0.0))
    centroid_distance = float(base.get("thermal_support_centroid_distance_px", 0.0))
    candidate_support_overlap = float(raw.get("support_thermal_overlap", 0.0))
    candidate_depth_overlap = float(raw.get("thermal_depth_overlap", 0.0))

    if gate_pixels < int(args.v9_min_gate_pixels):
        return f"v9 gate pixels {gate_pixels} below {int(args.v9_min_gate_pixels)}", "v9_gate_block"
    if gate_pixels > int(args.v9_max_gate_pixels):
        return f"v9 gate pixels {gate_pixels} exceed {int(args.v9_max_gate_pixels)}", "v9_gate_block"

    if mode == "v9_support_gated":
        if scene != "reflection_background":
            return "v9 support-gated mode is reserved for reflection/background scenes", "v9_reflection_block"
        if base_rejected < int(args.v9_reflection_min_rejected_pixels):
            return "v9 reflection gate lacks enough rejected background evidence", "v9_reflection_block"
        if base_score < float(args.v9_reflection_min_raw_score):
            return "v9 reflection gate raw score is not high enough to justify background suppression", "v9_reflection_block"
        if gate_pixels < int(args.v9_reflection_min_gate_pixels):
            return "v9 reflection gate is too small for the reflection/background case", "v9_reflection_block"
        if raw_lwir_edges < int(args.v9_reflection_min_lwir_edges):
            return "v9 reflection support-gated candidate has too few retained LWIR edges", "v9_reflection_block"
        return "", "v9_reflection_support_gated"

    if mode == "v9_depth_thermal":
        if scene == "reflection_background":
            if base_rejected < int(args.v9_reflection_min_rejected_pixels):
                return "v9 reflection depth-thermal lacks enough rejected background evidence", "v9_reflection_depth_block"
            if base_score < float(args.v9_reflection_min_raw_score):
                return "v9 reflection depth-thermal raw score is not high enough", "v9_reflection_depth_block"
            if raw_lwir_edges < int(args.v9_reflection_depth_min_lwir_edges):
                return "v9 reflection depth-thermal retains too few LWIR edges", "v9_reflection_depth_block"
            if candidate_depth_overlap < float(args.v9_min_thermal_depth_overlap):
                return "v9 reflection depth-thermal lacks thermal-depth overlap", "v9_reflection_depth_block"
            return "", "v9_reflection_depth_thermal"
        if scene != "double_edge_mismatch":
            return "v9 depth-thermal mode is reserved for double-edge mismatch scenes", "v9_depth_thermal_block"
        if base_score < float(args.v9_depth_thermal_min_raw_score):
            return "v9 depth-thermal gate raw score is not high enough", "v9_depth_thermal_block"
        if float(base.get("ghost_edge_ratio", 0.0)) < float(args.v9_depth_thermal_min_ghost_ratio):
            return "v9 depth-thermal lacks double-edge/ghost risk", "v9_depth_thermal_block"
        if base_thermal < int(args.v9_depth_thermal_min_baseline_thermal_pixels):
            return "v9 depth-thermal baseline thermal target is too small", "v9_depth_thermal_block"
        if not (int(args.v9_depth_thermal_min_lwir_edges) <= raw_lwir_edges <= int(args.v9_depth_thermal_max_lwir_edges)):
            return "v9 depth-thermal LWIR edge count outside reliable range", "v9_depth_thermal_block"
        if candidate_support_overlap < float(args.v9_depth_thermal_min_support_overlap):
            return "v9 depth-thermal lacks foreground support overlap", "v9_depth_thermal_block"
        if candidate_depth_overlap < float(args.v9_min_thermal_depth_overlap):
            return "v9 depth-thermal lacks thermal-depth overlap", "v9_depth_thermal_block"
        if float(raw.get("target_local_symmetric_edge_distance", 12.0)) > float(args.v9_depth_thermal_max_target_edge):
            return "v9 depth-thermal target-local edge remains too large", "v9_depth_thermal_block"
        return "", "v9_depth_thermal_double_edge"

    if mode == "v9_target_only":
        if scene != "general":
            return "v9 target-only mode is reserved for general weak-target scenes", "v9_target_only_block"
        if base_thermal > int(args.v9_weak_target_max_thermal_pixels):
            return "v9 target-only blocked because baseline thermal target is not weak/small", "v9_target_only_block"
        if centroid_distance < float(args.v9_min_centroid_distance_px):
            return "v9 target-only lacks thermal/support centroid disagreement", "v9_target_only_block"
        if not (int(args.v9_target_min_lwir_edges) <= raw_lwir_edges <= int(args.v9_target_max_lwir_edges)):
            return "v9 target-only LWIR edge count outside reliable range", "v9_target_only_block"
        if candidate_depth_overlap < float(args.v9_min_thermal_depth_overlap):
            return "v9 target-only lacks thermal-depth overlap", "v9_target_only_block"
        return "", "v9_weak_target_only"

    if mode == "v9_target_silhouette":
        if scene != "general":
            return "v9 silhouette mode is reserved for general target-contamination scenes", "v9_silhouette_block"
        if base_thermal < int(args.v9_silhouette_min_thermal_pixels):
            return "v9 silhouette blocked because baseline thermal target is too small", "v9_silhouette_block"
        if centroid_distance < float(args.v9_silhouette_min_centroid_distance_px):
            return "v9 silhouette lacks thermal/support centroid disagreement", "v9_silhouette_block"
        if not (int(args.v9_silhouette_min_lwir_edges) <= raw_lwir_edges <= int(args.v9_silhouette_max_lwir_edges)):
            return "v9 silhouette LWIR edge count outside reliable range", "v9_silhouette_block"
        if candidate_support_overlap < float(args.v9_silhouette_min_support_overlap):
            return "v9 silhouette lacks foreground support overlap", "v9_silhouette_block"
        if candidate_depth_overlap < float(args.v9_min_thermal_depth_overlap):
            return "v9 silhouette lacks thermal-depth overlap", "v9_silhouette_block"
        return "", "v9_target_silhouette"

    return "v9 support candidate mode is diagnostic-only unless a dedicated raw/depth gate allows it", "v9_mode_block"


def selection_adjusted_score_v6(record: dict, baseline: dict, args, rule: str) -> float:
    if rule.startswith("v9_"):
        raw = record["raw_metrics"]
        priority = {
            "v9_reflection_depth_thermal": -1.0,
            "v9_reflection_support_gated": 0.0,
            "v9_weak_target_only": 1.0,
            "v9_depth_thermal_double_edge": 1.5,
            "v9_target_silhouette": 2.0,
        }.get(rule, 10.0)
        edge_count = int(raw.get("raw_lwir_edge_pixels", 0))
        gate_pixels = int(record["candidate"].metadata.get("phase29_v9_gate_pixels", 0) or 0)
        return priority + abs(edge_count - float(args.v9_preferred_lwir_edges)) * 0.002 + gate_pixels * 0.00001
    return selection_adjusted_score_v5(record, baseline, args, rule)


def select_candidate_v1(records: list[dict], args) -> dict:
    baseline = next(record for record in records if record["candidate"].name == "p29_baseline")
    baseline_score = float(baseline["raw_metrics"]["raw_score"])
    rejected: list[str] = []
    eligible: list[dict] = []
    for record in records:
        if record is baseline:
            continue
        reason = safety_rejection(record, baseline, args)
        if reason:
            rejected.append(f"{record['candidate'].name}: {reason}")
            record["selector_debug"] = selector_debug_row(record, baseline, False, reason, selection_adjusted_score(record, baseline, args), "v1_block")
        else:
            eligible.append(record)
            record["selector_debug"] = selector_debug_row(record, baseline, True, "", selection_adjusted_score(record, baseline, args), "v1")

    if eligible:
        selected = min(eligible, key=lambda record: selection_adjusted_score(record, baseline, args))
        improvement = baseline_score - float(selected["raw_metrics"]["raw_score"])
        selected["selection_reason"] = (
            f"selected {selection_family(selected)}; raw_score improved by {improvement:.4f} vs baseline; "
            f"adjusted score {selection_adjusted_score(selected, baseline, args):.4f}"
        )
    else:
        selected = baseline
        best_raw = min((record for record in records if record is not baseline), key=lambda record: float(record["raw_metrics"]["raw_score"]))
        improvement = baseline_score - float(best_raw["raw_metrics"]["raw_score"])
        detail = "; ".join(rejected[:3])
        if len(rejected) > 3:
            detail += f"; +{len(rejected) - 3} more rejected"
        selected["selection_reason"] = (
            f"baseline retained; best raw improvement {improvement:.4f}; safety filters rejected candidates"
            + (f" ({detail})" if detail else "")
        )
    baseline["selector_debug"] = selector_debug_row(baseline, baseline, selected is baseline, "", selection_adjusted_score(baseline, baseline, args), "v1_baseline")
    return selected


def select_candidate_v2(records: list[dict], args) -> dict:
    baseline = next(record for record in records if record["candidate"].name == "p29_baseline")
    baseline_score = float(baseline["raw_metrics"]["raw_score"])
    rejected: list[str] = []
    eligible: list[tuple[dict, str, float]] = []
    for record in records:
        if record is baseline:
            continue
        reason, rule = safety_rejection_v2(record, baseline, args)
        adjusted = selection_adjusted_score_v2(record, baseline, args, rule)
        record["selector_debug"] = selector_debug_row(record, baseline, not bool(reason), reason, adjusted, rule)
        if reason:
            rejected.append(f"{record['candidate'].name}: {reason}")
        else:
            eligible.append((record, rule, adjusted))

    if eligible:
        selected, selected_rule, adjusted_score = min(eligible, key=lambda item: item[2])
        improvement = baseline_score - float(selected["raw_metrics"]["raw_score"])
        selected["selection_reason"] = (
            f"selector v2 selected {selection_family(selected)} via {selected_rule}; "
            f"raw_score improved by {improvement:.4f} vs baseline; adjusted score {adjusted_score:.4f}"
        )
    else:
        selected = baseline
        best_raw = min((record for record in records if record is not baseline), key=lambda record: float(record["raw_metrics"]["raw_score"]))
        improvement = baseline_score - float(best_raw["raw_metrics"]["raw_score"])
        detail = "; ".join(rejected[:3])
        if len(rejected) > 3:
            detail += f"; +{len(rejected) - 3} more rejected"
        selected["selection_reason"] = (
            f"selector v2 retained baseline; best raw improvement {improvement:.4f}; safety filters rejected candidates"
            + (f" ({detail})" if detail else "")
        )
    baseline["selector_debug"] = selector_debug_row(
        baseline,
        baseline,
        selected is baseline,
        "",
        selection_adjusted_score_v2(baseline, baseline, args, "baseline"),
        "v2_baseline",
    )
    return selected


def select_candidate_v3(records: list[dict], args) -> dict:
    selected = select_candidate_v2(records, args)
    gate = str(getattr(args, "reliability_gate", "strict"))
    selected["selection_reason"] = f"selector v3 reliability-gated ({gate}); {selected.get('selection_reason', '')}"
    for record in records:
        debug = record.setdefault("selector_debug", {})
        debug["phase29_selector_version_effective"] = "v3"
        debug["phase29_selector_reliability_gate"] = gate
        debug["phase29_selector_v5_note"] = (
            "v3 preserves the v2 raw-only safe selection and adds v5 reliability/ceiling evidence; "
            "v5 probe candidates are evaluation-only ceiling probes unless explicitly allowed as diagnostic guards."
        )
    return selected


def select_candidate_v4(records: list[dict], args) -> dict:
    baseline = next(record for record in records if record["candidate"].name == "p29_baseline")
    baseline_score = float(baseline["raw_metrics"]["raw_score"])
    rejected: list[str] = []
    eligible: list[tuple[dict, str, float]] = []
    for record in records:
        if record is baseline:
            continue
        reason, rule = safety_rejection_v4(record, baseline, args)
        adjusted = selection_adjusted_score_v4(record, baseline, args, rule)
        record["selector_debug"] = selector_debug_row(record, baseline, not bool(reason), reason, adjusted, rule)
        record["selector_debug"]["phase29_selector_version_effective"] = "v4"
        record["selector_debug"]["phase29_selector_reliability_gate"] = str(getattr(args, "reliability_gate", "strict"))
        record["selector_debug"]["phase29_selector_v6_note"] = (
            "v4 adds wide but raw/depth-gated risk-shift selection for v6; aligned metrics remain evaluation-only."
        )
        if reason:
            rejected.append(f"{record['candidate'].name}: {reason}")
        else:
            eligible.append((record, rule, adjusted))

    if eligible:
        selected, selected_rule, adjusted_score = min(eligible, key=lambda item: item[2])
        improvement = baseline_score - float(selected["raw_metrics"]["raw_score"])
        selected["selection_reason"] = (
            f"selector v4 selected {selection_family(selected)} via {selected_rule}; "
            f"raw_score improved by {improvement:.4f} vs baseline; adjusted score {adjusted_score:.4f}"
        )
    else:
        selected = baseline
        best_raw = min((record for record in records if record is not baseline), key=lambda record: float(record["raw_metrics"]["raw_score"]))
        improvement = baseline_score - float(best_raw["raw_metrics"]["raw_score"])
        detail = "; ".join(rejected[:3])
        if len(rejected) > 3:
            detail += f"; +{len(rejected) - 3} more rejected"
        selected["selection_reason"] = (
            f"selector v4 retained baseline; best raw improvement {improvement:.4f}; safety filters rejected candidates"
            + (f" ({detail})" if detail else "")
        )
    baseline["selector_debug"] = selector_debug_row(
        baseline,
        baseline,
        selected is baseline,
        "",
        selection_adjusted_score_v4(baseline, baseline, args, "baseline"),
        "v4_baseline",
    )
    baseline["selector_debug"]["phase29_selector_version_effective"] = "v4"
    baseline["selector_debug"]["phase29_selector_reliability_gate"] = str(getattr(args, "reliability_gate", "strict"))
    baseline["selector_debug"]["phase29_selector_v6_note"] = "v4 baseline row; aligned metrics are not used for selection."
    return selected


def select_candidate_v5(records: list[dict], args) -> dict:
    baseline = next(record for record in records if record["candidate"].name == "p29_baseline")
    baseline_score = float(baseline["raw_metrics"]["raw_score"])
    rejected: list[str] = []
    eligible: list[tuple[dict, str, float]] = []
    for record in records:
        if record is baseline:
            continue
        reason, rule = safety_rejection_v5(record, baseline, args)
        adjusted = selection_adjusted_score_v5(record, baseline, args, rule)
        record["selector_debug"] = selector_debug_row(record, baseline, not bool(reason), reason, adjusted, rule)
        record["selector_debug"]["phase29_selector_version_effective"] = "v5"
        record["selector_debug"]["phase29_selector_reliability_gate"] = str(getattr(args, "reliability_gate", "strict"))
        record["selector_debug"]["phase29_selector_v7_note"] = (
            "v5 adds target-local edge and support-stability gates for v7; aligned metrics remain evaluation-only."
        )
        if reason:
            rejected.append(f"{record['candidate'].name}: {reason}")
        else:
            eligible.append((record, rule, adjusted))

    if eligible:
        selected, selected_rule, adjusted_score = min(eligible, key=lambda item: item[2])
        improvement = baseline_score - float(selected["raw_metrics"]["raw_score"])
        target_gain = float(baseline["raw_metrics"].get("target_local_symmetric_edge_distance", 12.0)) - float(
            selected["raw_metrics"].get("target_local_symmetric_edge_distance", 12.0)
        )
        selected["selection_reason"] = (
            f"selector v5 selected {selection_family(selected)} via {selected_rule}; "
            f"raw_score delta {-improvement:+.4f}, target-local edge gain {target_gain:.4f}; "
            f"adjusted score {adjusted_score:.4f}"
        )
    else:
        selected = baseline
        best_raw = min((record for record in records if record is not baseline), key=lambda record: float(record["raw_metrics"]["raw_score"]))
        improvement = baseline_score - float(best_raw["raw_metrics"]["raw_score"])
        detail = "; ".join(rejected[:3])
        if len(rejected) > 3:
            detail += f"; +{len(rejected) - 3} more rejected"
        selected["selection_reason"] = (
            f"selector v5 retained baseline; best raw improvement {improvement:.4f}; safety filters rejected candidates"
            + (f" ({detail})" if detail else "")
        )
    baseline["selector_debug"] = selector_debug_row(
        baseline,
        baseline,
        selected is baseline,
        "",
        selection_adjusted_score_v5(baseline, baseline, args, "baseline"),
        "v5_baseline",
    )
    baseline["selector_debug"]["phase29_selector_version_effective"] = "v5"
    baseline["selector_debug"]["phase29_selector_reliability_gate"] = str(getattr(args, "reliability_gate", "strict"))
    baseline["selector_debug"]["phase29_selector_v7_note"] = "v5 baseline row; aligned metrics are not used for selection."
    return selected


def select_candidate_v6(records: list[dict], args) -> dict:
    baseline = next(record for record in records if record["candidate"].name == "p29_baseline")
    baseline_score = float(baseline["raw_metrics"]["raw_score"])
    rejected: list[str] = []
    eligible: list[tuple[dict, str, float]] = []
    for record in records:
        if record is baseline:
            continue
        reason, rule = safety_rejection_v6(record, baseline, args)
        adjusted = selection_adjusted_score_v6(record, baseline, args, rule)
        record["selector_debug"] = selector_debug_row(record, baseline, not bool(reason), reason, adjusted, rule)
        record["selector_debug"]["phase29_selector_version_effective"] = "v6"
        record["selector_debug"]["phase29_selector_reliability_gate"] = str(getattr(args, "reliability_gate", "strict"))
        record["selector_debug"]["phase29_selector_v9_note"] = (
            "v6 adds strict raw/depth support-gated selection for v9; aligned metrics remain evaluation-only."
        )
        if reason:
            rejected.append(f"{record['candidate'].name}: {reason}")
        else:
            eligible.append((record, rule, adjusted))

    if eligible:
        selected, selected_rule, adjusted_score = min(eligible, key=lambda item: item[2])
        improvement = baseline_score - float(selected["raw_metrics"]["raw_score"])
        selected["selection_reason"] = (
            f"selector v6 selected {selection_family(selected)} via {selected_rule}; "
            f"raw_score delta {-improvement:+.4f}; adjusted score {adjusted_score:.4f}; "
            "selection used raw RGB/LWIR/depth support gates only"
        )
    else:
        selected = baseline
        best_raw = min((record for record in records if record is not baseline), key=lambda record: float(record["raw_metrics"]["raw_score"]))
        improvement = baseline_score - float(best_raw["raw_metrics"]["raw_score"])
        detail = "; ".join(rejected[:3])
        if len(rejected) > 3:
            detail += f"; +{len(rejected) - 3} more rejected"
        selected["selection_reason"] = (
            f"selector v6 retained baseline; best raw improvement {improvement:.4f}; safety filters rejected candidates"
            + (f" ({detail})" if detail else "")
        )
    baseline["selector_debug"] = selector_debug_row(
        baseline,
        baseline,
        selected is baseline,
        "",
        selection_adjusted_score_v6(baseline, baseline, args, "baseline"),
        "v6_baseline",
    )
    baseline["selector_debug"]["phase29_selector_version_effective"] = "v6"
    baseline["selector_debug"]["phase29_selector_reliability_gate"] = str(getattr(args, "reliability_gate", "strict"))
    baseline["selector_debug"]["phase29_selector_v9_note"] = "v6 baseline row; aligned metrics are not used for selection."
    return selected


def select_candidate(records: list[dict], args) -> dict:
    if str(args.selector_version) == "v1":
        return select_candidate_v1(records, args)
    if str(args.selector_version) == "v3":
        return select_candidate_v3(records, args)
    if str(args.selector_version) == "v4":
        return select_candidate_v4(records, args)
    if str(args.selector_version) == "v5":
        return select_candidate_v5(records, args)
    if str(args.selector_version) == "v6":
        return select_candidate_v6(records, args)
    return select_candidate_v2(records, args)


def add_phase29_metadata(metrics: dict, candidate: CandidateOutput, raw_metrics: dict, selected: bool, scene: str, reason: str) -> dict:
    out = add_metadata_to_metrics(metrics, candidate)
    for key, value in candidate.metadata.items():
        if key.startswith("phase29") or key in {
            "allowed_for_generation",
            "depth_registration_dx_px",
            "depth_registration_dy_px",
            "stabilization_kernel_px",
            "fill_depth_border",
            "uses_aligned_for_generation",
            "anti_ghost_feather_px",
            "anti_ghost_erode_px",
            "max_support_area_frac",
            "calibration_file",
            "thermal_camera_calibration_file",
            "target_size",
        }:
            out[key] = value
    out.update(raw_metrics)
    out["phase29_scene_class"] = scene
    out["phase29_selected"] = bool(selected)
    out["phase29_selection_reason"] = reason
    out["phase29_selection_uses_aligned"] = False
    return out


def metric_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def reliability_label_color(label: str) -> tuple[int, int, int]:
    colors = {
        "accepted": (55, 180, 90),
        "improved": (35, 205, 205),
        "risky-pass": (40, 150, 245),
        "hard-ceiling-fail": (55, 55, 235),
        "selector-gap-fail": (190, 90, 235),
    }
    return colors.get(str(label), (180, 180, 180))


def add_reliability_fields(row: dict, args) -> None:
    edge = metric_float(row, "eval_lwir_to_mm5_aligned_t16_edge_distance", float("inf"))
    target = float(args.edge_target_px)
    selected_pass = bool(row.get("phase29_sample_pass"))
    oracle_pass = bool(row.get("phase29_oracle_sample_pass"))
    improvement = metric_float(row, "phase29_edge_improvement_vs_baseline", 0.0)
    oracle_gap = metric_float(row, "phase29_oracle_improvement_vs_selected", 0.0)
    ghost_ratio = metric_float(row, "ghost_edge_ratio", 0.0)
    tear_pixels = metric_float(row, "tear_risk_pixels", 0.0)
    rejected = metric_float(row, "rejected_background_pixels", 0.0)
    support_pixels = metric_float(row, "support_pixels", 0.0)
    support_compactness = metric_float(row, "support_compactness", 0.0)
    gate = str(getattr(args, "reliability_gate", "strict"))
    edge_slack = 0.25 if gate == "strict" else 0.12
    ghost_threshold = 0.90 if gate == "strict" else 0.95
    tear_threshold = 3500.0 if gate == "strict" else 4800.0
    support_threshold = 3500.0 if gate == "strict" else 2500.0

    hard_ceiling = bool((not selected_pass) and (not oracle_pass))
    selector_gap = bool((not selected_pass) and oracle_pass)
    risk_terms: list[str] = []
    if selected_pass:
        if edge >= target - edge_slack:
            risk_terms.append("edge close to threshold")
        if ghost_ratio >= ghost_threshold:
            risk_terms.append("double-edge/ghost risk")
        if tear_pixels >= tear_threshold:
            risk_terms.append("tear-risk concentration")
        if support_pixels < support_threshold:
            risk_terms.append("weak target support")
        if rejected >= 3000:
            risk_terms.append("background/reflection rejection is high")
        if support_compactness < 0.45:
            risk_terms.append("support is fragmented")
        if risk_terms:
            label = "risky-pass"
        elif improvement > 0.05:
            label = "improved"
        else:
            label = "accepted"
    elif hard_ceiling:
        label = "hard-ceiling-fail"
        risk_terms.append("best strict candidate is still above threshold")
    else:
        label = "selector-gap-fail"
        risk_terms.append("a strict candidate can pass, but raw-only selection did not safely choose it")

    score = 100.0
    score -= min(42.0, max(0.0, edge - 1.5) * 15.0)
    score -= min(18.0, ghost_ratio * 12.0)
    score -= min(14.0, tear_pixels / 650.0)
    score -= min(10.0, rejected / 1200.0)
    if support_pixels < support_threshold:
        score -= min(12.0, (support_threshold - support_pixels) / 180.0)
    if support_compactness < 0.55:
        score -= (0.55 - support_compactness) * 14.0
    if oracle_gap > 0.25 and not selected_pass:
        score -= min(12.0, oracle_gap * 4.0)
    if label == "hard-ceiling-fail":
        score = min(score, 35.0)
    elif label == "selector-gap-fail":
        score = min(score, 50.0)
    elif label == "risky-pass":
        score = min(score, 78.0)
    elif label == "improved":
        score = max(score, 74.0)
    score = max(0.0, min(100.0, score))

    if not risk_terms:
        reason = "raw/depth evidence is consistent and selected edge is comfortably inside the threshold"
    else:
        reason = "; ".join(risk_terms[:4])
    row["phase29_reliability_label"] = label
    row["phase29_reliability_score"] = float(score)
    row["phase29_reliability_reason"] = reason
    row["phase29_hard_ceiling"] = hard_ceiling
    row["phase29_selector_gap"] = selector_gap
    row["phase29_version_label"] = str(getattr(args, "version_label", ""))
    row["phase29_explainability_level"] = str(getattr(args, "explainability_level", ""))
    row["phase29_reliability_gate"] = gate
    row["phase29_failure_focus"] = str(getattr(args, "failure_focus", ""))


def reliability_map_view(
    rgb_bgr: np.ndarray,
    risk_view: np.ndarray,
    alpha: np.ndarray,
    focus_mask: np.ndarray,
    row: dict,
) -> np.ndarray:
    if risk_view.ndim == 2:
        risk_bgr = cv2.cvtColor(risk_view, cv2.COLOR_GRAY2BGR)
    else:
        risk_bgr = risk_view.copy()
    rgb = rgb_bgr.copy()
    if rgb.shape[:2] != risk_bgr.shape[:2]:
        rgb = cv2.resize(rgb, (risk_bgr.shape[1], risk_bgr.shape[0]), interpolation=cv2.INTER_AREA)
    out = cv2.addWeighted(rgb, 0.35, risk_bgr, 0.65, 0)
    label = str(row.get("phase29_reliability_label", "unknown"))
    color = np.array(reliability_label_color(label), dtype=np.uint8)
    mask = (focus_mask.astype(bool) | (alpha > 0.05))
    overlay = out.copy()
    overlay[mask] = color
    out = cv2.addWeighted(out, 0.68, overlay, 0.32, 0)
    cv2.rectangle(out, (0, 0), (out.shape[1], 42), (12, 12, 12), thickness=-1)
    cv2.putText(
        out,
        f"{label} score {metric_float(row, 'phase29_reliability_score', 0.0):.1f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        tuple(int(x) for x in color),
        2,
        cv2.LINE_AA,
    )
    return out


def write_failure_explanation(output_dir: Path, item: dict, row: dict, selected: dict, oracle_record: dict) -> None:
    if str(row.get("phase29_failure_focus", "all")) == "remaining" and bool(row.get("phase29_sample_pass")):
        return
    sample_tag = f"s{int(item['row'].aligned_id):03d}"
    oracle_edge = metric_float(row, "phase29_oracle_edge_distance", float("nan"))
    selected_edge = metric_float(row, "eval_lwir_to_mm5_aligned_t16_edge_distance", float("nan"))
    lines = [
        f"# Phase29 v5 Explanation: {row.get('sample_id', sample_tag)}",
        "",
        "## Boundary",
        "- Generation and raw selection use calibration files, raw RGB, raw LWIR, and raw depth only.",
        "- MM5 aligned data is used after generation only for evaluation and ceiling reporting.",
        "",
        "## Selected Result",
        f"- selected candidate: `{row.get('candidate', '')}`",
        f"- reliability: `{row.get('phase29_reliability_label', '')}` / `{metric_float(row, 'phase29_reliability_score', 0.0):.1f}`",
        f"- selected edge: `{selected_edge:.4f}px`",
        f"- baseline edge: `{metric_float(row, 'baseline_eval_lwir_to_mm5_aligned_t16_edge_distance', float('nan')):.4f}px`",
        f"- strict ceiling: `{row.get('phase29_oracle_candidate', '')}` / `{oracle_edge:.4f}px`",
        f"- scene class: `{row.get('phase29_scene_class', '')}`",
        f"- selector reason: `{row.get('phase29_selection_reason', '')}`",
        f"- reliability reason: `{row.get('phase29_reliability_reason', '')}`",
        "",
        "## Raw Evidence",
        f"- raw score: `{metric_float(row, 'raw_score', float('nan')):.4f}`",
        f"- support pixels: `{int(metric_float(row, 'support_pixels', 0.0))}`",
        f"- rejected background pixels: `{int(metric_float(row, 'rejected_background_pixels', 0.0))}`",
        f"- tear risk pixels: `{int(metric_float(row, 'tear_risk_pixels', 0.0))}`",
        f"- ghost edge ratio: `{metric_float(row, 'ghost_edge_ratio', 0.0):.4f}`",
        f"- thermal-depth centroid distance: `{metric_float(row, 'thermal_support_centroid_distance_px', 0.0):.4f}px`",
        "",
        "## Ceiling Interpretation",
    ]
    if bool(row.get("phase29_hard_ceiling")):
        lines.append("- Current strict candidate pool still cannot pass this sample; this is reported as a candidate-pool limit.")
    elif bool(row.get("phase29_selector_gap")):
        lines.append("- A generated strict candidate can pass, but raw-only evidence was not strong enough for the selector to choose it safely.")
    elif bool(row.get("phase29_sample_pass")):
        lines.append("- The selected strict result passes the edge threshold; reliability label describes residual visual risk.")
    else:
        lines.append("- The sample remains failed under the current strict-selected result.")
    lines.extend(
        [
            "",
            "## Visual Files",
            f"- acceptance summary: `acceptance_summary_panels/p29_{sample_tag}_acceptance_summary.png`",
            f"- hard ceiling panel: `hard_ceiling_panels/p29_{sample_tag}_hard_ceiling.png`",
            f"- reliability map: `reliability_maps/p29_{sample_tag}_reliability.png`",
            f"- candidate panel: `candidate_panels/p29_{sample_tag}_candidate_panel.png`",
        ]
    )
    (output_dir / "failure_explanations" / f"p29_{sample_tag}_explanation.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_visual_outputs(output_dir: Path, item: dict, baseline: dict, selected: dict, selected_metrics: dict, args) -> None:
    row = item["row"]
    sample_tag = f"s{int(row.aligned_id):03d}"
    candidate = selected["candidate"]
    context = selected["context"]
    baseline_candidate = baseline["candidate"]
    baseline_context = baseline["context"]
    valid = context["valid"]
    focus_mask = context["focus_mask"]
    alpha = context["alpha"]
    depth_crop = context["depth_crop"]
    depth_valid = context["depth_valid"]
    rejected_mask = context["rejected_mask"]
    support = context["support"]

    depth_view = depth_support_view(depth_crop, depth_valid, support, valid)
    clean_depth_view = depth_support_view(depth_crop, depth_valid, focus_mask, valid)
    clean_lwir_view = clean_registered_lwir_view(candidate.lwir, candidate.lwir_valid, focus_mask)
    alpha_view = alpha_mask_view(alpha, focus_mask, rejected_mask)
    anti_blend = deghost_acceptance_fusion(candidate.rgb, candidate.lwir, valid, focus_mask, alpha)
    contour_overlay = contour_alignment_overlay(candidate.rgb, candidate.lwir, valid, focus_mask, depth_crop, depth_valid)
    risk_view = context["risk_view"]
    error_heat = edge_error_heatmap(item["aligned_lwir_u8"], candidate.lwir, candidate.lwir_valid)
    reliability_view = reliability_map_view(candidate.rgb, risk_view, alpha, focus_mask, selected_metrics)
    baseline_alpha = baseline_context["alpha"]
    baseline_blend = deghost_acceptance_fusion(
        baseline_candidate.rgb,
        baseline_candidate.lwir,
        baseline_context["valid"],
        baseline_context["focus_mask"],
        baseline_alpha,
    )
    baseline_risk = baseline_context["risk_view"]

    imwrite_unicode(output_dir / "selected_registered_lwir" / f"p29_{sample_tag}_selected_lwir.png", candidate.lwir)
    imwrite_unicode(output_dir / "selected_clean_lwir" / f"p29_{sample_tag}_clean_lwir.png", clean_lwir_view)
    imwrite_unicode(output_dir / "selected_fusion_review" / f"p29_{sample_tag}_fusion_review.png", anti_blend)
    imwrite_unicode(output_dir / "selection_maps" / f"p29_{sample_tag}_selection_alpha.png", alpha_view)
    imwrite_unicode(output_dir / "contour_overlays" / f"p29_{sample_tag}_contours.png", contour_overlay)
    imwrite_unicode(output_dir / "tear_ghost_maps" / f"p29_{sample_tag}_tear_ghost_risk.png", risk_view)
    imwrite_unicode(output_dir / "edge_error_heatmaps" / f"p29_{sample_tag}_edge_error.png", error_heat)
    imwrite_unicode(output_dir / "reliability_maps" / f"p29_{sample_tag}_reliability.png", reliability_view)

    make_six_panel(
        [
            (candidate.rgb, "Generated RGB"),
            (clean_depth_view, "P29 selected support"),
            (clean_lwir_view, "P29 clean LWIR"),
            (anti_blend, "P29 fusion"),
            (contour_overlay, f"Contours {selected_metrics['eval_lwir_to_mm5_aligned_t16_edge_distance']:.3f}px"),
            (risk_view, str(selected_metrics.get("phase29_scene_class", "scene"))),
        ],
        output_dir / "selected_acceptance_panels" / f"p29_{sample_tag}_acceptance.png",
    )
    make_panel_grid(
        [
            (candidate.rgb, "Generated RGB"),
            (clean_lwir_view, f"Selected LWIR {selected_metrics['eval_lwir_to_mm5_aligned_t16_edge_distance']:.3f}px"),
            (contour_overlay, "RGB/LWIR/Depth contours"),
            (anti_blend, "Anti-ghost fusion"),
            (risk_view, "Tear/Ghost/Edge risk"),
        ],
        output_dir / "five_panels" / f"p29_{sample_tag}_five_panel.png",
        tile_size=(260, 205),
        columns=5,
    )
    make_panel_grid(
        [
            (contour_overlay, f"Selected edge {selected_metrics['eval_lwir_to_mm5_aligned_t16_edge_distance']:.3f}px"),
            (clean_lwir_view, f"Target raw {selected_metrics.get('target_local_symmetric_edge_distance', float('nan')):.2f}"),
            (risk_view, f"Ghost {selected_metrics.get('ghost_edge_ratio', 0.0):.3f}"),
            (alpha_view, f"Support {int(selected_metrics.get('support_pixels', 0))}"),
            (error_heat, "Eval-only edge heat"),
        ],
        output_dir / "edge_acceptance_panels" / f"p29_{sample_tag}_edge_acceptance.png",
        tile_size=(300, 225),
        columns=5,
    )
    if bool(getattr(args, "save_edge_debug", False)):
        make_panel_grid(
            [
                (baseline_blend, f"Baseline {baseline['raw_metrics']['raw_score']:.2f}"),
                (anti_blend, f"Selected {selected['raw_metrics']['raw_score']:.2f}"),
                (contour_overlay, f"Target edge {selected['raw_metrics'].get('target_local_symmetric_edge_distance', float('nan')):.2f}"),
                (risk_view, f"P90 {selected['raw_metrics'].get('target_local_edge_p90', float('nan')):.2f}"),
                (reliability_view, str(selected_metrics.get("phase29_selector_rule", ""))),
            ],
            output_dir / "edge_debug_panels" / f"p29_{sample_tag}_edge_debug.png",
            tile_size=(300, 225),
            columns=5,
        )

    make_panel_grid(
        [
            (baseline_blend, "P28 baseline fusion"),
            (anti_blend, "P29 selected fusion"),
            (baseline_risk, f"P28 risk score {baseline['raw_metrics']['raw_score']:.2f}"),
            (risk_view, f"P29 raw score {selected['raw_metrics']['raw_score']:.2f}"),
            (depth_view, "Raw depth support"),
            (alpha_view, str(candidate.name)),
        ],
        output_dir / "before_after_phase28_phase29" / f"p29_{sample_tag}_before_after.png",
        tile_size=(330, 245),
        columns=3,
    )

    ranked_records = sorted(selected["all_records"], key=lambda record: float(record["raw_metrics"]["raw_score"]))
    panel_records: list[dict] = []
    for record in ranked_records[:4]:
        panel_records.append(record)
    oracle_record = min(
        selected["all_records"],
        key=lambda record: float(record.get("eval_metrics", {}).get("eval_lwir_to_mm5_aligned_t16_edge_distance", float("inf"))),
    )
    for record in (selected, baseline, oracle_record):
        if record not in panel_records:
            panel_records.append(record)
    top_records = panel_records[: min(6, len(panel_records))]
    panels = []
    for record in top_records:
        cand = record["candidate"]
        ctx = record["context"]
        blend = deghost_acceptance_fusion(cand.rgb, cand.lwir, ctx["valid"], ctx["focus_mask"], ctx["alpha"])
        edge = record.get("eval_metrics", {}).get("eval_lwir_to_mm5_aligned_t16_edge_distance", float("nan"))
        panels.append((blend, f"{cand.name} raw {record['raw_metrics']['raw_score']:.2f} eval {edge:.2f}"))
    make_panel_grid(
        panels,
        output_dir / "candidate_panels" / f"p29_{sample_tag}_candidate_panel.png",
        tile_size=(330, 245),
        columns=3,
    )

    oracle = oracle_record["candidate"]
    oracle_context = oracle_record["context"]
    oracle_alpha = oracle_context["alpha"]
    oracle_focus = oracle_context["focus_mask"]
    oracle_depth = oracle_context["depth_crop"]
    oracle_depth_valid = oracle_context["depth_valid"]
    oracle_blend = deghost_acceptance_fusion(oracle.rgb, oracle.lwir, oracle_context["valid"], oracle_focus, oracle_alpha)
    oracle_clean = clean_registered_lwir_view(oracle.lwir, oracle.lwir_valid, oracle_focus)
    oracle_contours = contour_alignment_overlay(oracle.rgb, oracle.lwir, oracle_context["valid"], oracle_focus, oracle_depth, oracle_depth_valid)
    oracle_risk = oracle_context["risk_view"]
    oracle_edge = oracle_record.get("eval_metrics", {}).get("eval_lwir_to_mm5_aligned_t16_edge_distance", float("nan"))
    make_panel_grid(
        [
            (anti_blend, f"Selected {candidate.name}"),
            (oracle_blend, f"Oracle ceiling {oracle.name}"),
            (clean_lwir_view, f"Selected edge {selected_metrics['eval_lwir_to_mm5_aligned_t16_edge_distance']:.3f}px"),
            (oracle_clean, f"Ceiling edge {oracle_edge:.3f}px"),
            (contour_overlay, "Selected contours"),
            (oracle_contours, "Ceiling contours eval-only"),
            (risk_view, "Selected risk"),
            (oracle_risk, "Ceiling risk"),
        ],
        output_dir / "oracle_ceiling_panels" / f"p29_{sample_tag}_oracle_ceiling.png",
        tile_size=(330, 245),
        columns=4,
    )
    reliability_label = str(selected_metrics.get("phase29_reliability_label", "unknown"))
    pass_label = "PASS" if bool(selected_metrics.get("phase29_sample_pass")) else "FAIL"
    hard_label = "HARD CEILING" if bool(selected_metrics.get("phase29_hard_ceiling")) else "SELECTOR/SELECTED"
    make_panel_grid(
        [
            (anti_blend, f"Selected {pass_label} {selected_metrics['eval_lwir_to_mm5_aligned_t16_edge_distance']:.3f}px"),
            (oracle_blend, f"Strict ceiling {oracle_edge:.3f}px"),
            (contour_overlay, "Selected RGB/LWIR/Depth contours"),
            (oracle_contours, "Ceiling contours eval-only"),
            (reliability_view, f"{reliability_label} {selected_metrics.get('phase29_reliability_score', 0.0):.1f}"),
            (risk_view, "Tear/Ghost risk"),
            (alpha_view, "Support alpha/rejection"),
            (error_heat, "Eval edge heatmap"),
        ],
        output_dir / "acceptance_summary_panels" / f"p29_{sample_tag}_acceptance_summary.png",
        tile_size=(330, 245),
        columns=4,
    )
    make_panel_grid(
        [
            (anti_blend, f"Selected {candidate.name}"),
            (oracle_blend, f"Best strict {oracle.name}"),
            (contour_overlay, f"Selected {selected_metrics['eval_lwir_to_mm5_aligned_t16_edge_distance']:.3f}px"),
            (oracle_contours, f"Ceiling {oracle_edge:.3f}px"),
            (risk_view, str(selected_metrics.get("phase29_scene_class", "scene"))),
            (reliability_view, hard_label),
        ],
        output_dir / "hard_ceiling_panels" / f"p29_{sample_tag}_hard_ceiling.png",
        tile_size=(330, 245),
        columns=3,
    )
    write_failure_explanation(output_dir, item, selected_metrics, selected, oracle_record)

    crop_mask = focus_mask if np.any(focus_mask) else valid
    crop = foreground_crop(crop_mask, margin=96)
    if crop is not None:
        x0, y0, x1, y1 = crop
        make_six_panel(
            [
                (candidate.rgb[y0:y1, x0:x1], "RGB ROI"),
                (clean_depth_view[y0:y1, x0:x1], "Support ROI"),
                (clean_lwir_view[y0:y1, x0:x1], "LWIR ROI"),
                (anti_blend[y0:y1, x0:x1], "Fusion ROI"),
                (contour_overlay[y0:y1, x0:x1], "Contour ROI"),
                (risk_view[y0:y1, x0:x1], "Risk ROI"),
            ],
            output_dir / "roi_panels" / f"p29_{sample_tag}_roi.png",
            tile_size=(260, 205),
        )


def make_output_dirs(output_dir: Path) -> None:
    for child in (
        "metrics",
        "reports",
        "selected_registered_lwir",
        "selected_clean_lwir",
        "selected_fusion_review",
        "selected_acceptance_panels",
        "roi_panels",
        "before_after_phase28_phase29",
        "candidate_panels",
        "oracle_ceiling_panels",
        "acceptance_summary_panels",
        "five_panels",
        "edge_acceptance_panels",
        "edge_debug_panels",
        "hard_ceiling_panels",
        "reliability_maps",
        "failure_explanations",
        "selector_debug",
        "selection_maps",
        "contour_overlays",
        "tear_ghost_maps",
        "edge_error_heatmaps",
    ):
        (output_dir / child).mkdir(parents=True, exist_ok=True)


def summarize_selected(rows: list[dict]) -> list[dict]:
    summary = summarize(rows)
    if not summary:
        return []
    best = summary[0]
    numeric_keys = [
        "eval_lwir_to_mm5_aligned_t16_edge_distance",
        "eval_lwir_to_mm5_aligned_t16_ncc",
        "eval_rgb_to_mm5_aligned_rgb_ncc",
        "raw_score",
        "raw_symmetric_edge_distance",
        "target_local_symmetric_edge_distance",
        "target_local_edge_p90",
        "edge_stability_score",
        "tear_risk_pixels",
        "ghost_edge_ratio",
        "depth_discontinuity_pixels",
        "support_pixels",
        "rejected_background_pixels",
        "baseline_eval_lwir_to_mm5_aligned_t16_edge_distance",
        "phase29_edge_improvement_vs_baseline",
        "phase29_oracle_edge_distance",
        "phase29_oracle_improvement_vs_selected",
        "phase29_oracle_improvement_vs_baseline",
        "phase29_selector_adjusted_score",
        "phase29_selector_raw_improvement",
        "phase29_selector_edge_growth",
        "phase29_selector_support_ratio",
        "phase29_selector_shift_l1",
        "phase29_selector_target_edge_gain",
        "phase29_selector_target_p90_gain",
        "phase29_selector_context_edge_delta",
        "phase29_reliability_score",
    ]
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
        if values:
            best[f"{key}_mean"] = float(np.mean(values))
            best[f"{key}_min"] = float(np.min(values))
            best[f"{key}_max"] = float(np.max(values))
    best["phase29_sample_pass_count"] = int(sum(1 for row in rows if bool(row.get("phase29_sample_pass"))))
    best["phase29_sample_fail_count"] = int(sum(1 for row in rows if not bool(row.get("phase29_sample_pass"))))
    best["phase29_improved_count"] = int(sum(1 for row in rows if float(row.get("phase29_edge_improvement_vs_baseline", 0.0)) > 0.0))
    best["phase29_regressed_count"] = int(sum(1 for row in rows if float(row.get("phase29_edge_improvement_vs_baseline", 0.0)) < -1e-6))
    best["phase29_oracle_sample_pass_count"] = int(sum(1 for row in rows if bool(row.get("phase29_oracle_sample_pass"))))
    best["phase29_oracle_sample_fail_count"] = int(sum(1 for row in rows if not bool(row.get("phase29_oracle_sample_pass"))))
    labels = ["accepted", "improved", "risky-pass", "hard-ceiling-fail", "selector-gap-fail"]
    for label in labels:
        best[f"phase29_reliability_{label.replace('-', '_')}_count"] = int(
            sum(1 for row in rows if str(row.get("phase29_reliability_label", "")) == label)
        )
    best["phase29_selection_uses_aligned"] = False
    return [best]


def ceiling_reason(row: dict) -> str:
    scene = str(row.get("phase29_scene_class", "general"))
    if bool(row.get("phase29_sample_pass")):
        return "passed"
    if scene == "reflection_background":
        return "remaining failure is dominated by reflection/background thermal evidence or abnormal depth support"
    if scene == "weak_target_support":
        return "remaining failure has too little reliable target support for raw-only local selection"
    if scene == "depth_tearing":
        return "remaining failure has depth discontinuity near the support/fusion boundary"
    if scene == "double_edge_mismatch":
        return "remaining failure has conflicting RGB/LWIR edge evidence under the strict raw-only score"
    return "remaining failure did not receive enough raw-only evidence to justify moving away from the safe candidate"


def write_report(output_dir: Path, payload: dict, selected_rows: list[dict], candidate_rows: list[dict]) -> None:
    best = payload["phase29_candidate"]
    failed = [row for row in selected_rows if not bool(row.get("phase29_sample_pass"))]
    improved = [row for row in selected_rows if float(row.get("phase29_edge_improvement_vs_baseline", 0.0)) > 0.0]
    oracle_unsolved = [
        row for row in selected_rows
        if not bool(row.get("phase29_oracle_sample_pass")) and not bool(row.get("phase29_sample_pass"))
    ]
    selector_gap = [
        row for row in selected_rows
        if bool(row.get("phase29_oracle_sample_pass")) and not bool(row.get("phase29_sample_pass"))
    ]
    lines = [
        "# Phase29 Broad-Generalization Report",
        "",
        "## Boundary",
        "- Generation uses calibration files, raw RGB/LWIR, raw depth, and calibration-derived helpers.",
        "- MM5 aligned RGB/T16 are evaluation-only.",
        "- Candidate selection uses raw-only reliability scores and does not read aligned metrics.",
        "",
        "## Profile",
        f"- profile: `{payload['profile']}`",
        f"- aligned ids: `{payload['aligned_ids']}`",
        f"- selector version: `{payload.get('selector_version', '')}`",
        f"- candidate grid: `{payload.get('candidate_grid', '')}`",
        f"- run mode: `{payload.get('run_mode', '')}`",
        f"- report level: `{payload.get('report_level', '')}`",
        f"- version label: `{payload.get('version_label', '')}`",
        f"- explainability level: `{payload.get('explainability_level', '')}`",
        f"- reliability gate: `{payload.get('reliability_gate', '')}`",
        f"- failure focus: `{payload.get('failure_focus', '')}`",
        "",
        "## Result",
        f"- selected edge mean/max: `{best.get('eval_lwir_to_mm5_aligned_t16_edge_distance_mean', float('nan')):.4f}` / `{best.get('eval_lwir_to_mm5_aligned_t16_edge_distance_max', float('nan')):.4f}` px",
        f"- selected LWIR NCC mean/min: `{best.get('eval_lwir_to_mm5_aligned_t16_ncc_mean', float('nan')):.4f}` / `{best.get('eval_lwir_to_mm5_aligned_t16_ncc_min', float('nan')):.4f}`",
        f"- pass/fail: `{best.get('phase29_sample_pass_count', 0)}` / `{best.get('phase29_sample_fail_count', 0)}`",
        f"- improved/regressed vs Phase28 baseline: `{best.get('phase29_improved_count', 0)}` / `{best.get('phase29_regressed_count', 0)}`",
        f"- raw score mean/min/max: `{best.get('raw_score_mean', float('nan')):.4f}` / `{best.get('raw_score_min', float('nan')):.4f}` / `{best.get('raw_score_max', float('nan')):.4f}`",
        f"- reliability score mean/min: `{best.get('phase29_reliability_score_mean', float('nan')):.2f}` / `{best.get('phase29_reliability_score_min', float('nan')):.2f}`",
        "- reliability labels: "
        f"`accepted={best.get('phase29_reliability_accepted_count', 0)}`, "
        f"`improved={best.get('phase29_reliability_improved_count', 0)}`, "
        f"`risky-pass={best.get('phase29_reliability_risky_pass_count', 0)}`, "
        f"`hard-ceiling-fail={best.get('phase29_reliability_hard_ceiling_fail_count', 0)}`, "
        f"`selector-gap-fail={best.get('phase29_reliability_selector_gap_fail_count', 0)}`",
        "",
        "## Evaluation-Only Candidate Ceiling",
        "- This ceiling is computed after every strict candidate is generated; it is not used for generation or raw-only selection.",
        f"- ceiling edge mean/max: `{best.get('phase29_oracle_edge_distance_mean', float('nan')):.4f}` / `{best.get('phase29_oracle_edge_distance_max', float('nan')):.4f}` px",
        f"- ceiling pass/fail: `{best.get('phase29_oracle_sample_pass_count', 0)}` / `{best.get('phase29_oracle_sample_fail_count', 0)}`",
        f"- candidate-pool-unsolved failures: `{len(oracle_unsolved)}`",
        f"- raw-selector gap failures: `{len(selector_gap)}`",
        "",
        "## Per-Sample Selection",
    ]
    for row in selected_rows:
        lines.append(
            "- `{}`: selected `{}`, reliability `{}` `{:.1f}`, scene `{}`, raw `{:.3f}`, edge `{:.4f}px`, baseline `{:.4f}px`, delta `{:+.4f}px`, oracle `{}` `{:.4f}px`, reason `{}`".format(
                row.get("sample_id", ""),
                row.get("candidate", ""),
                row.get("phase29_reliability_label", ""),
                float(row.get("phase29_reliability_score", float("nan"))),
                row.get("phase29_scene_class", ""),
                float(row.get("raw_score", float("nan"))),
                float(row.get("eval_lwir_to_mm5_aligned_t16_edge_distance", float("nan"))),
                float(row.get("baseline_eval_lwir_to_mm5_aligned_t16_edge_distance", float("nan"))),
                float(row.get("phase29_edge_improvement_vs_baseline", float("nan"))),
                row.get("phase29_oracle_candidate", ""),
                float(row.get("phase29_oracle_edge_distance", float("nan"))),
                row.get("phase29_selection_reason", ""),
            )
        )
    if str(payload.get("report_level", "acceptance")) == "research":
        lines.extend(["", "## Selector Debug Summary"])
        for row in selected_rows:
            lines.append(
                "- `{}`: rule `{}`, eligible `{}`, adjusted `{:.4f}`, raw_improvement `{:.4f}`, edge_growth `{:.3f}`, shift `({:+.2f},{:+.2f})`".format(
                    row.get("sample_id", ""),
                    row.get("phase29_selector_rule", ""),
                    row.get("phase29_selector_eligible", ""),
                    float(row.get("phase29_selector_adjusted_score", float("nan"))),
                    float(row.get("phase29_selector_raw_improvement", float("nan"))),
                    float(row.get("phase29_selector_edge_growth", float("nan"))),
                    float(row.get("phase29_selector_shift_delta_dx", float("nan"))),
                    float(row.get("phase29_selector_shift_delta_dy", float("nan"))),
                )
            )
    lines.extend(["", "## Remaining Failures"])
    if not failed:
        lines.append("- none")
    else:
        for row in failed:
            lines.append(
                "- `{}`: edge `{:.4f}px`; reliability `{}`; ceiling reason: {}".format(
                    row.get("sample_id", ""),
                    float(row.get("eval_lwir_to_mm5_aligned_t16_edge_distance", float("nan"))),
                    row.get("phase29_reliability_label", ""),
                    ceiling_reason(row),
                )
            )
    lines.extend(["", "## Hard-Ceiling Interpretation"])
    if oracle_unsolved:
        lines.append("- Candidate pool still cannot pass these selected failures under the current strict generated candidates:")
        for row in oracle_unsolved:
            lines.append(
                "  - `{}`: selected `{:.4f}px`, best strict candidate `{}` `{:.4f}px`".format(
                    row.get("sample_id", ""),
                    float(row.get("eval_lwir_to_mm5_aligned_t16_edge_distance", float("nan"))),
                    row.get("phase29_oracle_candidate", ""),
                    float(row.get("phase29_oracle_edge_distance", float("nan"))),
                )
            )
    else:
        lines.append("- No selected failure is blocked by the current strict candidate pool.")
    if selector_gap:
        lines.append("- These failures have an evaluation-only passing candidate, but raw-only selection did not safely choose it:")
        for row in selector_gap:
            lines.append(
                "  - `{}`: selected `{}` `{:.4f}px`, ceiling `{}` `{:.4f}px`".format(
                    row.get("sample_id", ""),
                    row.get("candidate", ""),
                    float(row.get("eval_lwir_to_mm5_aligned_t16_edge_distance", float("nan"))),
                    row.get("phase29_oracle_candidate", ""),
                    float(row.get("phase29_oracle_edge_distance", float("nan"))),
                )
            )
    else:
        lines.append("- No remaining failure has a hidden passing strict candidate in the current pool.")
    lines.extend(
        [
            "",
            "## Candidate Evidence",
            f"- total candidate rows: `{len(candidate_rows)}`",
            "- `metrics/p29_candidates.csv` contains raw-only scores and aligned-only post-selection evaluation columns.",
            "- `candidate_panels/` shows the top raw-scored candidates per sample for audit.",
            "- `oracle_ceiling_panels/` compares the selected result with the evaluation-only best strict candidate.",
            "- `acceptance_summary_panels/` is the v5 one-page visual acceptance evidence for every sample.",
            "- `hard_ceiling_panels/` shows selected-vs-ceiling evidence for hard-limit review.",
            "- `reliability_maps/` overlays selected support and risk using the v5 reliability label.",
            "- `failure_explanations/` contains per-sample Markdown explanations.",
            "- `selector_debug/` is written when `--save-selector-debug` is enabled.",
            "- `before_after_phase28_phase29/` compares the Phase28 baseline and Phase29 selected result.",
        ]
    )
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports" / "p29_broad_generalization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_selector_debug(output_dir: Path, item: dict, records: list[dict]) -> None:
    row = item["row"]
    sample_tag = f"s{int(row.aligned_id):03d}"
    debug_rows = []
    for record in records:
        out = {
            "sample_id": sample_id(row),
            "candidate": record["candidate"].name,
            "raw_score": float(record["raw_metrics"].get("raw_score", float("nan"))),
            "raw_symmetric_edge_distance": float(record["raw_metrics"].get("raw_symmetric_edge_distance", float("nan"))),
            "eval_edge": float(record.get("eval_metrics", {}).get("eval_lwir_to_mm5_aligned_t16_edge_distance", float("nan"))),
        }
        out.update(record.get("selector_debug", {}))
        debug_rows.append(out)
    write_csv(output_dir / "selector_debug" / f"p29_{sample_tag}_selector_debug.csv", debug_rows, collect_fieldnames(debug_rows, []))


def process_item(item: dict, args, specs: list[CandidateSpec]) -> tuple[dict, list[dict], list[dict]]:
    records: list[dict] = []
    for spec in specs:
        candidate = make_phase29_candidate(item, args, spec)
        context = focus_context(item, candidate, args)
        raw_metrics = raw_selection_score(item, candidate, context, args)
        records.append({"candidate": candidate, "context": context, "raw_metrics": raw_metrics})

    selected = select_candidate(records, args)
    selected["all_records"] = records
    baseline = next(record for record in records if record["candidate"].name == "p29_baseline")
    baseline_scene = scene_class(baseline["raw_metrics"])

    candidate_rows: list[dict] = []
    for record in records:
        candidate = record["candidate"]
        scene = scene_class(record["raw_metrics"]) if candidate.name == "p29_baseline" else baseline_scene
        eval_metrics = evaluate_candidate(item["row"], candidate, item["aligned_rgb"], item["aligned_lwir_u8"])
        record["eval_metrics"] = eval_metrics
        row = add_phase29_metadata(
            eval_metrics,
            candidate,
            record["raw_metrics"],
            selected=(record is selected),
            scene=scene,
            reason=selected.get("selection_reason", "") if record is selected else "",
        )
        row.update(record.get("selector_debug", {}))
        row["sample_id"] = sample_id(item["row"])
        candidate_rows.append(row)

    if bool(args.save_selector_debug):
        write_selector_debug(args.current_output_dir, item, records)

    selected_eval = selected["eval_metrics"]
    baseline_eval = baseline["eval_metrics"]
    selected_scene = scene_class(baseline["raw_metrics"])
    selected_row = add_phase29_metadata(
        selected_eval,
        selected["candidate"],
        selected["raw_metrics"],
        selected=True,
        scene=selected_scene,
        reason=selected.get("selection_reason", ""),
    )
    selected_row["sample_id"] = sample_id(item["row"])
    selected_row["baseline_candidate"] = baseline["candidate"].name
    selected_row["baseline_raw_score"] = float(baseline["raw_metrics"]["raw_score"])
    selected_row["baseline_eval_lwir_to_mm5_aligned_t16_edge_distance"] = float(
        baseline_eval["eval_lwir_to_mm5_aligned_t16_edge_distance"]
    )
    selected_row["phase29_edge_improvement_vs_baseline"] = float(
        baseline_eval["eval_lwir_to_mm5_aligned_t16_edge_distance"]
        - selected_eval["eval_lwir_to_mm5_aligned_t16_edge_distance"]
    )
    selected_row["phase29_sample_pass"] = bool(
        float(selected_eval["eval_lwir_to_mm5_aligned_t16_edge_distance"]) < float(args.edge_target_px)
        and str(selected_row.get("uses_aligned_for_generation", "true")).lower() == "false"
    )
    selected_row.update(selected.get("selector_debug", {}))
    oracle_row = min(
        candidate_rows,
        key=lambda row: float(row.get("eval_lwir_to_mm5_aligned_t16_edge_distance", float("inf"))),
    )
    oracle_edge = float(oracle_row.get("eval_lwir_to_mm5_aligned_t16_edge_distance", float("nan")))
    selected_edge = float(selected_eval["eval_lwir_to_mm5_aligned_t16_edge_distance"])
    baseline_edge = float(baseline_eval["eval_lwir_to_mm5_aligned_t16_edge_distance"])
    selected_row["phase29_oracle_candidate"] = oracle_row.get("candidate", "")
    selected_row["phase29_oracle_edge_distance"] = oracle_edge
    selected_row["phase29_oracle_sample_pass"] = bool(
        oracle_edge < float(args.edge_target_px)
        and str(oracle_row.get("uses_aligned_for_generation", "true")).lower() == "false"
    )
    selected_row["phase29_oracle_improvement_vs_selected"] = float(selected_edge - oracle_edge)
    selected_row["phase29_oracle_improvement_vs_baseline"] = float(baseline_edge - oracle_edge)
    selected_row["phase29_oracle_selection_uses_aligned"] = True
    selected_row["phase29_oracle_note"] = "evaluation-only candidate ceiling; not used for generation or raw-only selection"
    add_reliability_fields(selected_row, args)
    add_target_metrics(item, selected["candidate"], selected_row, selected["context"]["focus_mask"])
    write_visual_outputs(args.current_output_dir, item, baseline, selected, selected_row, args)
    return selected_row, candidate_rows, records


def run_profile(args, profile: str, output_dir: Path) -> dict:
    local_args = copy.copy(args)
    local_args.current_profile = profile
    local_args.current_output_dir = output_dir
    if not str(local_args.aligned_ids).strip():
        local_args.aligned_ids = PROFILE_IDS[profile]
    make_output_dirs(output_dir)

    prepared_rows, context = prepare_rows(local_args)
    specs = candidate_specs(local_args)
    selected_rows: list[dict] = []
    candidate_rows: list[dict] = []
    for item in prepared_rows:
        print(f"processing {sample_id(item['row'])} phase29 candidate set [{profile}]")
        selected_row, item_candidate_rows, _records = process_item(item, local_args, specs)
        selected_rows.append(selected_row)
        candidate_rows.extend(item_candidate_rows)

    summary_rows = summarize_selected(selected_rows)
    best = summary_rows[0] if summary_rows else {}
    metric_gate_passed = (
        bool(summary_rows)
        and float(best.get("eval_lwir_to_mm5_aligned_t16_edge_distance_mean", float("inf"))) < float(local_args.edge_target_px)
        and float(best.get("eval_lwir_to_mm5_aligned_t16_edge_distance_max", float("inf"))) < float(local_args.edge_target_px)
        and int(best.get("phase29_sample_fail_count", 1)) == 0
    )
    metric_gate_required = profile in {"core", "review"}

    write_csv(output_dir / "metrics" / "p29_metrics.csv", selected_rows, collect_fieldnames(selected_rows, []))
    write_csv(output_dir / "metrics" / "p29_candidates.csv", candidate_rows, collect_fieldnames(candidate_rows, []))
    write_csv(output_dir / "metrics" / "p29_summary.csv", summary_rows, collect_fieldnames(summary_rows, []))
    write_csv(output_dir / "metrics" / "p29_v5_reliability.csv", selected_rows, collect_fieldnames(selected_rows, []))
    payload = {
        "profile": profile,
        "aligned_ids": str(local_args.aligned_ids),
        "context": context,
        "candidate_count_per_sample": len(specs),
        "edge_target_px": float(local_args.edge_target_px),
        "metric_gate_required": bool(metric_gate_required),
        "metric_gate_passed": bool(metric_gate_passed),
        "required_acceptance_passed": bool(metric_gate_passed or not metric_gate_required),
        "aligned_usage": "evaluation_only_not_generation_or_selection",
        "selector_version": str(local_args.selector_version),
        "candidate_grid": str(local_args.candidate_grid),
        "run_mode": str(local_args.run_mode),
        "report_level": str(local_args.report_level),
        "version_label": str(getattr(local_args, "version_label", "")),
        "explainability_level": str(getattr(local_args, "explainability_level", "")),
        "reliability_gate": str(getattr(local_args, "reliability_gate", "")),
        "failure_focus": str(getattr(local_args, "failure_focus", "")),
        "phase29_candidate": best,
        "candidate_summary": summary_rows,
        "per_sample": selected_rows,
    }
    (output_dir / "metrics" / "p29_best.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_report(output_dir, payload, selected_rows, candidate_rows)
    print(f"Phase29 {profile} output: {output_dir}")
    if best:
        print(
            "edge mean/max: "
            f"{best.get('eval_lwir_to_mm5_aligned_t16_edge_distance_mean', float('nan')):.4f} / "
            f"{best.get('eval_lwir_to_mm5_aligned_t16_edge_distance_max', float('nan')):.4f} px"
        )
        print(f"sample pass/fail: {best.get('phase29_sample_pass_count')} / {best.get('phase29_sample_fail_count')}")
        print(f"improved/regressed: {best.get('phase29_improved_count')} / {best.get('phase29_regressed_count')}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase29 broad-generalization registration with raw-only candidate selection.")
    parser.add_argument("--index", default="mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv")
    parser.add_argument("--aligned-ids", default="")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--splits", default="test,val")
    parser.add_argument("--calibration", default="calibration/def_stereocalib_THERM.yml")
    parser.add_argument("--thermal-camera-calibration", default="calibration/def_thermalcam_ori.yml")
    parser.add_argument("--calibration-root", default="")
    parser.add_argument("--target-size", default="640x480")
    parser.add_argument("--max-board-offset-rmse-px", type=float, default=12.0)
    parser.add_argument("--point-margin-px", type=int, default=80)
    parser.add_argument("--ransac-threshold-px", type=float, default=4.0)
    parser.add_argument("--dx-px", type=float, default=3.5)
    parser.add_argument("--dy-px", type=float, default=2.0)
    parser.add_argument("--stabilization-kernel-px", type=int, default=9)
    parser.add_argument("--edge-target-px", type=float, default=3.0)
    parser.add_argument("--anti-ghost-feather-px", type=float, default=3.0)
    parser.add_argument("--anti-ghost-erode-px", type=int, default=2)
    parser.add_argument("--max-support-area-frac", type=float, default=0.10)
    parser.add_argument("--min-support-ratio", type=float, default=0.006)
    parser.add_argument("--max-raw-support-ratio", type=float, default=0.070)
    parser.add_argument("--selector-version", choices=["v1", "v2", "v3", "v4", "v5", "v6"], default="v4")
    parser.add_argument(
        "--candidate-grid",
        choices=["local", "wide-safe", "edge-v7", "edge-v7-compact", "acceptance-lite", "component-v8", "support-v9"],
        default="acceptance-lite",
    )
    parser.add_argument("--run-mode", choices=["strict-selected", "ceiling-study"], default="strict-selected")
    parser.add_argument("--report-level", choices=["acceptance", "research"], default="research")
    parser.add_argument("--version-label", default="v6")
    parser.add_argument("--explainability-level", choices=["compact", "full"], default="full")
    parser.add_argument("--reliability-gate", choices=["strict", "balanced"], default="strict")
    parser.add_argument("--failure-focus", choices=["all", "remaining"], default="all")
    parser.add_argument("--save-selector-debug", action="store_true")
    parser.add_argument("--raw-score-margin", type=float, default=0.60)
    parser.add_argument("--guard-score-margin", type=float, default=0.65)
    parser.add_argument("--weak-support-raw-score-margin", type=float, default=0.20)
    parser.add_argument("--weak-support-pixels", type=int, default=3000)
    parser.add_argument("--raw-edge-margin", type=float, default=0.035)
    parser.add_argument("--low-risk-raw-score", type=float, default=9.0)
    parser.add_argument("--max-lwir-edge-growth", type=float, default=1.12)
    parser.add_argument("--max-guard-lwir-edge-growth", type=float, default=1.08)
    parser.add_argument("--reflection-overlap-threshold", type=float, default=0.20)
    parser.add_argument("--min-lwir-edge-retain", type=float, default=0.55)
    parser.add_argument("--min-selection-support-retain", type=float, default=0.65)
    parser.add_argument("--max-selection-support-growth", type=float, default=1.18)
    parser.add_argument("--generic-max-shift-l1", type=float, default=2.0)
    parser.add_argument("--min-guard-baseline-support-pixels", type=int, default=4500)
    parser.add_argument("--shift-prior-weight", type=float, default=0.05)
    parser.add_argument("--guard-prior-penalty", type=float, default=0.22)
    parser.add_argument("--depth-tearing-raw-margin", type=float, default=0.10)
    parser.add_argument("--depth-tearing-raw-score-min", type=float, default=16.8)
    parser.add_argument("--depth-tearing-raw-score-max", type=float, default=17.4)
    parser.add_argument("--depth-tearing-raw-edge-slack", type=float, default=0.16)
    parser.add_argument("--depth-tearing-max-lwir-edge-growth", type=float, default=1.08)
    parser.add_argument("--depth-tearing-max-abs-shift", type=float, default=1.0)
    parser.add_argument("--depth-tearing-depth-edge-weight", type=float, default=0.8)
    parser.add_argument("--double-edge-raw-score-min", type=float, default=22.0)
    parser.add_argument("--double-edge-ghost-ratio-min", type=float, default=0.92)
    parser.add_argument("--double-edge-raw-margin", type=float, default=0.04)
    parser.add_argument("--double-edge-min-shift-l1", type=float, default=2.8)
    parser.add_argument("--double-edge-max-shift-l1", type=float, default=4.2)
    parser.add_argument("--double-edge-max-lwir-edge-growth", type=float, default=1.55)
    parser.add_argument("--double-edge-max-centroid-delta-px", type=float, default=8.0)
    parser.add_argument("--double-edge-edge-growth-weight", type=float, default=0.55)
    parser.add_argument("--double-edge-shift-weight", type=float, default=0.03)
    parser.add_argument("--v6-extended-raw-margin", type=float, default=0.35)
    parser.add_argument("--v6-low-risk-raw-margin", type=float, default=0.75)
    parser.add_argument("--v6-extended-raw-edge-margin", type=float, default=0.10)
    parser.add_argument("--v6-raw-edge-slack", type=float, default=0.20)
    parser.add_argument("--v6-min-ghost-reduction", type=float, default=0.006)
    parser.add_argument("--v6-min-tear-reduction-pixels", type=int, default=80)
    parser.add_argument("--v6-max-ghost-increase", type=float, default=0.025)
    parser.add_argument("--v6-high-risk-ghost-ratio", type=float, default=0.86)
    parser.add_argument("--v6-high-risk-raw-score", type=float, default=17.0)
    parser.add_argument("--v6-high-risk-rejected-pixels", type=int, default=2500)
    parser.add_argument("--v6-min-positive-delta-dx", type=float, default=0.5)
    parser.add_argument("--v6-edge-contamination-min-thermal-pixels", type=int, default=1600)
    parser.add_argument("--v6-edge-contamination-max-compactness", type=float, default=0.65)
    parser.add_argument("--v6-edge-contamination-max-ghost-ratio", type=float, default=0.86)
    parser.add_argument("--v6-edge-contamination-max-raw-score", type=float, default=15.0)
    parser.add_argument("--v6-edge-contamination-raw-margin", type=float, default=0.50)
    parser.add_argument("--v6-generic-max-shift-l1", type=float, default=4.2)
    parser.add_argument("--v6-high-risk-max-shift-l1", type=float, default=5.8)
    parser.add_argument("--v6-weak-support-max-shift-l1", type=float, default=5.2)
    parser.add_argument("--v6-max-abs-shift", type=float, default=4.0)
    parser.add_argument("--v6-min-support-retain", type=float, default=0.60)
    parser.add_argument("--v6-max-support-growth", type=float, default=1.25)
    parser.add_argument("--v6-max-lwir-edge-growth", type=float, default=1.65)
    parser.add_argument("--v6-shift-weight", type=float, default=0.12)
    parser.add_argument("--v6-edge-growth-weight", type=float, default=0.18)
    parser.add_argument("--v6-ghost-increase-weight", type=float, default=8.0)
    parser.add_argument("--v7-raw-score-slack", type=float, default=0.55)
    parser.add_argument("--v7-context-edge-slack", type=float, default=0.45)
    parser.add_argument("--v7-target-local-edge-margin", type=float, default=0.08)
    parser.add_argument("--v7-reflection-target-edge-margin", type=float, default=0.75)
    parser.add_argument("--v7-guard-target-edge-margin", type=float, default=0.08)
    parser.add_argument("--v7-target-p90-slack", type=float, default=0.35)
    parser.add_argument("--v7-min-ghost-reduction", type=float, default=0.004)
    parser.add_argument("--v7-min-tear-reduction-pixels", type=int, default=60)
    parser.add_argument("--v7-max-ghost-increase", type=float, default=0.030)
    parser.add_argument("--v7-edge-max-shift-l1", type=float, default=6.5)
    parser.add_argument("--v7-max-abs-shift", type=float, default=4.75)
    parser.add_argument("--v7-min-support-retain", type=float, default=0.45)
    parser.add_argument("--v7-max-support-growth", type=float, default=1.35)
    parser.add_argument("--v7-max-lwir-edge-growth", type=float, default=1.75)
    parser.add_argument("--v7-general-max-compactness", type=float, default=0.72)
    parser.add_argument("--v7-general-max-target-edge", type=float, default=5.10)
    parser.add_argument("--v7-general-max-target-p90", type=float, default=12.0)
    parser.add_argument("--v7-target-edge-weight", type=float, default=0.60)
    parser.add_argument("--v7-target-p90-weight", type=float, default=0.08)
    parser.add_argument("--v7-stability-weight", type=float, default=0.22)
    parser.add_argument("--v7-shift-weight", type=float, default=0.08)
    parser.add_argument("--v7-edge-growth-weight", type=float, default=0.16)
    parser.add_argument("--v7-ghost-increase-weight", type=float, default=7.0)
    parser.add_argument("--v8-component-max-shift-px", type=float, default=8.0)
    parser.add_argument("--v8-component-pair-max-distance-px", type=float, default=18.0)
    parser.add_argument("--v8-component-min-shift-l1", type=float, default=1.0)
    parser.add_argument("--v8-component-min-thermal-area", type=int, default=120)
    parser.add_argument("--v8-component-min-target-area", type=int, default=120)
    parser.add_argument("--v8-component-max-components", type=int, default=6)
    parser.add_argument("--v8-component-dilate-px", type=int, default=11)
    parser.add_argument("--v9-min-gate-pixels", type=int, default=700)
    parser.add_argument("--v9-max-gate-pixels", type=int, default=22000)
    parser.add_argument("--v9-reflection-min-rejected-pixels", type=int, default=3000)
    parser.add_argument("--v9-reflection-min-raw-score", type=float, default=18.0)
    parser.add_argument("--v9-reflection-min-gate-pixels", type=int, default=8000)
    parser.add_argument("--v9-reflection-min-lwir-edges", type=int, default=80)
    parser.add_argument("--v9-reflection-depth-min-lwir-edges", type=int, default=250)
    parser.add_argument("--v9-min-centroid-distance-px", type=float, default=60.0)
    parser.add_argument("--v9-min-thermal-depth-overlap", type=float, default=0.85)
    parser.add_argument("--v9-weak-target-max-thermal-pixels", type=int, default=1200)
    parser.add_argument("--v9-target-min-lwir-edges", type=int, default=80)
    parser.add_argument("--v9-target-max-lwir-edges", type=int, default=900)
    parser.add_argument("--v9-silhouette-min-thermal-pixels", type=int, default=1400)
    parser.add_argument("--v9-silhouette-min-centroid-distance-px", type=float, default=70.0)
    parser.add_argument("--v9-silhouette-min-lwir-edges", type=int, default=30)
    parser.add_argument("--v9-silhouette-max-lwir-edges", type=int, default=220)
    parser.add_argument("--v9-silhouette-min-support-overlap", type=float, default=0.55)
    parser.add_argument("--v9-depth-thermal-min-raw-score", type=float, default=22.0)
    parser.add_argument("--v9-depth-thermal-min-ghost-ratio", type=float, default=0.90)
    parser.add_argument("--v9-depth-thermal-min-baseline-thermal-pixels", type=int, default=3000)
    parser.add_argument("--v9-depth-thermal-min-lwir-edges", type=int, default=1000)
    parser.add_argument("--v9-depth-thermal-max-lwir-edges", type=int, default=2200)
    parser.add_argument("--v9-depth-thermal-min-support-overlap", type=float, default=0.55)
    parser.add_argument("--v9-depth-thermal-max-target-edge", type=float, default=5.0)
    parser.add_argument("--v9-preferred-lwir-edges", type=float, default=220.0)
    parser.add_argument("--include-nofill-shift-candidates", action="store_true")
    parser.add_argument("--allow-diagnostic-guard-selection", action="store_true")
    parser.add_argument("--save-edge-debug", action="store_true")
    parser.add_argument(
        "--shift-deltas",
        default="-1,-1;0,-1;1,-1;-1,0;1,0;-1,1;0,1;1,1;-0.5,0;0.5,0;0,-0.5;0,0.5",
    )
    parser.add_argument("--run-profile", choices=["core", "review", "broad", "all"], default="broad")
    parser.add_argument("--output", default=str(PHASE29_DIR / "outputs_broad_acceptance_lite"))
    args = parser.parse_args()

    if str(args.run_mode) == "ceiling-study":
        args.candidate_grid = "edge-v7" if str(getattr(args, "version_label", "")) == "v7" else "wide-safe"
        args.report_level = "research"
        args.save_selector_debug = True

    output_dir = Path(args.output)
    profiles = ["core", "review", "broad"] if args.run_profile == "all" and not str(args.aligned_ids).strip() else [
        args.run_profile if args.run_profile != "all" else "core"
    ]
    payloads = {}
    for profile in profiles:
        profile_output = output_dir / profile if len(profiles) > 1 else output_dir
        payloads[profile] = run_profile(args, profile, profile_output)
    if len(payloads) > 1:
        (output_dir / "p29_all_profiles_summary.json").write_text(
            json.dumps(payloads, ensure_ascii=False, indent=2, default=json_default),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
