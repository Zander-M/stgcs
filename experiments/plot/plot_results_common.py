from __future__ import annotations

import bisect
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np


class PlotPalette:
    INK = "#222222"
    BLACK = "#000000"
    NEUTRAL_DARK = "#6f6f6f"
    NEUTRAL_MID = "#777777"
    NEUTRAL_LIGHT = "#a7a7a7"
    WHITE = "white"
    BLUE = "blue"

    HEURISTIC_ZERO = NEUTRAL_LIGHT
    HEURISTIC_SC = "#4f8f5f"
    HEURISTIC_LBG = "#3f6fb5"
    HEURISTIC_TD = "#c45a45"
    HEURISTIC_MAX = BLACK

    DOMAIN_RANDOM = "#b6462a"
    DOMAIN_MAZE = INK
    DOMAIN_IRIS = "#1d5e8a"

    HEURISTIC_COLORS: Dict[str, str] = {
        "Zero": HEURISTIC_ZERO,
        "SC": HEURISTIC_SC,
        "LBG": HEURISTIC_LBG,
        "TD": HEURISTIC_TD,
        "Max": HEURISTIC_MAX,
    }
    DOMINATION_COLORS: Dict[str, str] = {
        "GUB": HEURISTIC_ZERO,
        "ESC": HEURISTIC_SC,
        "IPC": HEURISTIC_LBG,
        "ISC": HEURISTIC_TD,
    }
    DOMAIN_COLORS: Dict[str, str] = {
        "grid2d": DOMAIN_RANDOM,
        "rand": DOMAIN_RANDOM,
        "maze": DOMAIN_MAZE,
        "iris2d": DOMAIN_IRIS,
        "iris-2d": DOMAIN_IRIS,
        "iris": DOMAIN_IRIS,
    }
    ST_METHOD_COLORS: Dict[str, str] = {
        "ipc": INK,
        "esc": NEUTRAL_DARK,
        "esc_eps1": "#0072b2",
        "micp": "#d55e00",
        "micp_rounding": "#e69f00",
        "st_rrt_first": "#009e73",
        "st_rrt_final": "#5abf90",
        "zeta_sipp": "#cc79a7",
        "zeta_sipp_2r": "#9b5f8f",
    }
    MRMP_RULE_COLORS: Dict[str, str] = {
        "lazy": HEURISTIC_SC,
        "soc": HEURISTIC_LBG,
        "makespan": HEURISTIC_TD,
        "num_conflicts": HEURISTIC_MAX,
    }
    MRMP_WINDOW_COLORS: Dict[str, str] = {
        "pbs_nc": BLACK,
        "fixed": HEURISTIC_TD,
        "dynamic": HEURISTIC_SC,
        "dynamic_beta": HEURISTIC_LBG,
    }

    @classmethod
    def heuristic_color(cls, name: str) -> str:
        return cls.HEURISTIC_COLORS[name]

    @classmethod
    def domination_color(cls, name: str) -> str:
        return cls.DOMINATION_COLORS[name]

    @classmethod
    def domain_color(cls, name: str) -> str:
        return cls.DOMAIN_COLORS[name]


def dedupe_result_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    latest_by_key: Dict[tuple[str, float], Dict[str, object]] = {}
    for row in rows:
        key = (str(row["instance_id"]), float(row["budget"]))
        latest_by_key[key] = row
    return list(latest_by_key.values())


def _filter_rows(
    rows: Sequence[Dict[str, object]],
    budget: float | None = None,
    instance_ids: set[str] | None = None,
    runtime_threshold: float | None = None,
) -> List[Dict[str, object]]:
    filtered: List[Dict[str, object]] = []
    for row in rows:
        if budget is not None and not math.isclose(float(row["budget"]), budget):
            continue
        if instance_ids is not None and str(row["instance_id"]) not in instance_ids:
            continue
        if runtime_threshold is not None:
            runtime = float(row["runtime"])
            if not math.isfinite(runtime) or runtime > runtime_threshold:
                continue
        filtered.append(row)
    return filtered


class BudgetSuccessCurve:
    @staticmethod
    def thresholds(
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        planners: Sequence[str],
        field: str,
        max_budget: float | None = None,
    ) -> List[float]:
        success_values: List[float] = []
        observed_values: List[float] = []
        for planner in planners:
            for row in rows_by_planner.get(planner, []):
                value = float(row[field])
                if not math.isfinite(value) or value <= 0.0:
                    continue
                if max_budget is not None:
                    value = min(value, max_budget)
                observed_values.append(value)
                if bool(row["is_success"]):
                    success_values.append(value)

        right_anchor = max_budget if max_budget is not None else (max(observed_values) if observed_values else 1.0)
        left_anchor = max(right_anchor / 100.0, 1e-3)
        if not success_values:
            return [left_anchor, right_anchor]

        thresholds = sorted(set(success_values))
        first_value = thresholds[0]
        left_anchor = max(min(first_value / 2.0, right_anchor / 10.0), 1e-3)
        if left_anchor < first_value:
            thresholds = [left_anchor, *thresholds]
        if not math.isclose(thresholds[-1], right_anchor):
            thresholds.append(right_anchor)
        return thresholds

    @staticmethod
    def success_rates(rows: Sequence[Dict[str, object]], thresholds: Sequence[float], field: str) -> List[float]:
        if not thresholds:
            return []
        if not rows:
            return [math.nan] * len(thresholds)

        success_values = sorted(
            float(row[field])
            for row in rows
            if bool(row["is_success"]) and math.isfinite(float(row[field])) and float(row[field]) > 0.0
        )
        denominator = len(rows)
        return [bisect.bisect_right(success_values, threshold) / denominator for threshold in thresholds]


class BoxplotStyle:
    DEFAULT_WIDTH = 0.42
    DEFAULT_FACE_COLOR = "white"
    DEFAULT_WHIS = (5, 95)
    BOX_LINEWIDTH = 1.1
    MEDIAN_LINEWIDTH = 1.25
    WHISKER_LINEWIDTH = 0.9
    SCATTER_WIDTH_FRACTION = 0.55
    SCATTER_SIZE = 8.0
    SCATTER_ALPHA = 0.72

    @classmethod
    def draw(
        cls,
        ax: plt.Axes,
        values: Sequence[Sequence[float]],
        positions: Sequence[float],
        widths: float | Sequence[float],
        color: str,
        facecolor: str = DEFAULT_FACE_COLOR,
        showfliers: bool = False,
        whis: tuple[float, float] | None = DEFAULT_WHIS,
        orientation: str = "vertical",
    ) -> Dict[str, object]:
        kwargs: Dict[str, object] = {
            "orientation": orientation,
            "positions": positions,
            "widths": widths,
            "manage_ticks": False,
            "patch_artist": True,
            "showfliers": showfliers,
            "boxprops": {
                "facecolor": facecolor,
                "edgecolor": color,
                "linewidth": cls.BOX_LINEWIDTH,
            },
            "medianprops": {
                "color": color,
                "linewidth": cls.MEDIAN_LINEWIDTH,
            },
            "whiskerprops": {
                "color": color,
                "linewidth": cls.WHISKER_LINEWIDTH,
            },
            "capprops": {
                "color": color,
                "linewidth": cls.WHISKER_LINEWIDTH,
            },
        }
        if whis is not None:
            kwargs["whis"] = whis
        return ax.boxplot(values, **kwargs)

    @classmethod
    def scatter_points(
        cls,
        ax: plt.Axes,
        position: float,
        values: Sequence[float],
        width: float = DEFAULT_WIDTH,
        color: str = PlotPalette.INK,
        size: float = SCATTER_SIZE,
        alpha: float = SCATTER_ALPHA,
        orientation: str = "vertical",
    ) -> None:
        jittered_positions = cls.scatter_x_positions(position, width, len(values))
        if orientation == "vertical":
            ax.scatter(
                jittered_positions,
                values,
                s=size,
                marker="o",
                facecolors=color,
                edgecolors="white",
                linewidths=0.25,
                alpha=alpha,
                zorder=3,
            )
            return
        if orientation == "horizontal":
            ax.scatter(
                values,
                jittered_positions,
                s=size,
                marker="o",
                facecolors=color,
                edgecolors="white",
                linewidths=0.25,
                alpha=alpha,
                zorder=3,
            )
            return
        raise ValueError(f"Unsupported scatter orientation {orientation!r}.")

    @classmethod
    def scatter_x_positions(cls, position: float, width: float, count: int) -> list[float]:
        if count <= 1:
            return [float(position)] * count
        span = float(width) * cls.SCATTER_WIDTH_FRACTION
        return [
            float(position) + span * (idx / (count - 1) - 0.5)
            for idx in range(count)
        ]


def _apply_figure_header_layout(
    fig: plt.Figure,
    title: str,
    top: float = 0.84,
    w_pad: float = 0.15,
    h_pad: float = 0.45,
) -> None:
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top), pad=0.35, w_pad=w_pad, h_pad=h_pad)


def _set_square_cells(axes: np.ndarray) -> None:
    for ax in axes.flat:
        ax.set_box_aspect(1)


def _save_figure(fig: plt.Figure, output_prefix: Path, formats: Sequence[str]) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        output_path = output_prefix.with_suffix(f".{fmt}")
        fig.savefig(output_path, dpi=240, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    plt.close(fig)
