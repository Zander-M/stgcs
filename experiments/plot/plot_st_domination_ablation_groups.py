from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

from experiments.mpl_paper import configure_matplotlib_for_latex

configure_matplotlib_for_latex(backend="Agg")

from matplotlib.lines import Line2D
from matplotlib.ticker import (
    FixedLocator,
    FormatStrFormatter,
    LogFormatterMathtext,
    LogLocator,
    MaxNLocator,
    NullFormatter,
    PercentFormatter,
)

from experiments.st_runners.heuristic_ablation_st_manifest import STHeuristicAblationManifestBuilder
from experiments.plot.plot_results_common import BoxplotStyle, PlotPalette, _save_figure, plt
from experiments.plot.plot_st_heuristic_ablation_groups import STHeuristicAblationGroupReport


class STDominationAblationGroupReport(STHeuristicAblationGroupReport):
    DEFAULT_MANIFEST = Path("data/instances/st_planning/manifest.json")
    DEFAULT_RESULTS_ROOT = Path("data/results/st_planning/domination_ablation")
    DEFAULT_OUTPUT_PREFIX = Path("latex/figs/dc_ablation")

    PLANNERS = (
        "Search(Max+GUB)",
        "Search(Max+GUB+ESC)",
        "Search(Max+GUB+ISC)",
        "Search(Max+GUB+IPC)",
    )
    FIGURE_PLANNERS = PLANNERS
    QUALITY_FIGURE_PLANNERS = (
        "Search(Max+GUB+ISC)",
        "Search(Max+GUB+IPC)",
    )
    PLANNER_SHORT_LABELS = {
        "Search(Max+GUB)": "UB",
        "Search(Max+GUB+ESC)": r"$+\delta_\text{set}$",
        "Search(Max+GUB+IPC)": r"$+\delta_\text{pos}$",
        "Search(Max+GUB+ISC)": r"$+\delta_\text{state}$",
    }
    PAIRS = (
        ("Search(Max+GUB+ESC)", "Search(Max+GUB+IPC)"),
        ("Search(Max+GUB+ESC)", "Search(Max+GUB+ISC)"),
        ("Search(Max+GUB+IPC)", "Search(Max+GUB+ISC)"),
    )
    METRIC_FIELDS = {
        "expanded": "num_expanded_nodes",
        "runtime": "runtime",
        "quality": "cost",
    }
    METRIC_LABELS = {
        "expanded": "expanded nodes",
        "runtime": "runtime (s)",
        "quality": "relative cost increase",
    }
    FIGURE_X_LABELS = {
        "expanded": "expanded nodes",
        "runtime": "runtime (s)",
        "quality": "optimality gap",
    }
    X_LABELS = {
        "expanded": r"$\delta_\text{set}$ expanded nodes",
        "runtime": r"$\delta_\text{set}$ runtime (s)",
        "quality": r"$\delta_\text{set}$ solution cost",
    }
    Y_LABEL_FORMATS = {
        "expanded": "{} node reduction",
        "runtime": "{} runtime increase",
        "quality": "{} relative cost increase",
    }
    PERCENT_Y_METRICS = frozenset(("expanded", "runtime", "quality"))
    INCREASE_Y_METRICS = frozenset(("runtime", "quality"))
    INCREASE_AXIS_PADDING_FRACTION = 0.12
    QUALITY_Y_TICKS = (0.0, 0.2, 0.4)
    EXPANDED_NODE_GAP_Y_LABEL_COORDS = {
        "Search(Max+GUB+IPC)": (-0.15, 0.5),
        "Search(Max+GUB+ISC)": (-0.15, 0.5),
    }
    FIGURE_METRIC_ORDER = ("runtime", "expanded", "quality")
    FIGURE_GROUP_ORDER = ("grid2d", "maze", "iris2d")
    FIGURE_GROUP_LABELS = {
        "grid2d": "rand",
        "maze": "maze",
        "iris2d": "iris",
    }
    FIGURE_HEURISTIC_COLORS = {
        "Search(Max+GUB)": STHeuristicAblationGroupReport.FIGURE_HEURISTIC_COLORS[
            "Search(SC+GUB)"
        ],
        "Search(Max+GUB+ESC)": STHeuristicAblationGroupReport.FIGURE_HEURISTIC_COLORS[
            "Search(LBG+GUB)"
        ],
        "Search(Max+GUB+IPC)": STHeuristicAblationGroupReport.FIGURE_HEURISTIC_COLORS[
            "Search(Max+GUB)"
        ],
        "Search(Max+GUB+ISC)": STHeuristicAblationGroupReport.FIGURE_HEURISTIC_COLORS[
            "Search(TD+GUB)"
        ],
    }
    FIGURE_Y_TICKS = {
        "runtime": {
            "grid2d": (1e-1, 1e0, 1e1, 1e2),
            "maze": (1e-1, 1e0, 1e1, 1e2),
            "iris2d": (1e-1, 1e0, 1e1, 1e2),
        },
        "expanded": {
            "grid2d": (1e1, 1e2, 1e3),
            "maze": (1e1, 1e2, 1e3),
            "iris2d": (1e1, 1e2, 1e3),
        },
        "quality": {
            "grid2d": None,
            "maze": None,
            "iris2d": None,
        },
    }
    FIGURE_Y_LIMITS = {
        "runtime": {
            "grid2d": None,
            "maze": None,
            "iris2d": None,
        },
        "expanded": {
            "grid2d": (5, 2.5e3),
            "maze": (5, 2e3),
            "iris2d": (5, 2e3),
        },
        "quality": {
            "grid2d": None,
            "maze": None,
            "iris2d": None,
        },
    }
    FIGURE_Y_SCALES = {
        "runtime": "log",
        "expanded": "log",
        "quality": "linear",
    }
    FIGURE_BOXPLOT_WIDTH = 0.42
    FIGURE_ROW_HEIGHT_RATIOS = (1.0, 1.0, 0.62)
    FIGURE_SHOW_SCATTER_POINTS = True
    FIGURE_SCATTER_SIZE = 10
    FIGURE_SCATTER_ALPHA = 0.62
    FIGURE_BOXPLOT_ORIENTATION = "horizontal"
    QUALITY_ZERO_TOLERANCE = 1e-6
    QUALITY_Y_AXIS_MIN_HIGH = 0.02
    QUALITY_Y_AXIS_PADDING_FRACTION = 0.12
    INLINE_X_LABELS = frozenset((r"$\delta_\text{set}$", r"$\delta_\text{pos}$"))
    INLINE_X_LABEL_COORDS = (0.65, 0.12)
    INLINE_X_LABEL_FONTSIZE = 10.5
    FIGURE_SIZE = (6, 5)
    REFERENCE_PLANNER = "Search(Max+GUB+ESC)"
    GAP_PLANNERS = (
        "Search(Max+GUB+ISC)",
        "Search(Max+GUB+IPC)",
    )
    GROUP_ORDER = ("maze", "grid2d", "iris2d")
    GROUP_LABELS = {
        "maze": "maze",
        "grid2d": "grid2d",
        "iris2d": "Iris/iris2d",
    }
    FIGURE_GROUP_STYLES = {
        "grid2d": {
            "label": "rand",
            "marker": "^",
            "facecolor": "none",
            "edgecolor": PlotPalette.domain_color("grid2d"),
        },
        "maze": {
            "label": "maze",
            "marker": "o",
            "facecolor": "none",
            "edgecolor": PlotPalette.domain_color("maze"),
        },
        "iris2d": {
            "label": "iris",
            "marker": "s",
            "facecolor": "none",
            "edgecolor": PlotPalette.domain_color("iris2d"),
        },
    }

    @classmethod
    def _boxplot_values(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        group: str,
        budget: float,
        metric_field: str,
    ) -> list[list[float]]:
        if metric_field != "cost":
            return super()._boxplot_values(rows_by_planner, group, budget, metric_field)

        common_instance_ids = cls._common_success_instance_ids(
            rows_by_planner,
            group,
            budget,
            metric_field,
            (cls.REFERENCE_PLANNER, *cls.QUALITY_FIGURE_PLANNERS),
        )
        reference_rows = cls._success_rows_by_instance(
            cls._rows_for_group(rows_by_planner.get(cls.REFERENCE_PLANNER, []), group, budget),
            metric_field,
        )
        values_by_planner: list[list[float]] = []
        for planner in cls.QUALITY_FIGURE_PLANNERS:
            planner_rows = cls._success_rows_by_instance(
                cls._rows_for_group(rows_by_planner.get(planner, []), group, budget),
                metric_field,
            )
            values: list[float] = []
            for instance_id in sorted(common_instance_ids):
                reference_value = cls._finite_metric_value(reference_rows[instance_id], metric_field)
                planner_value = cls._finite_metric_value(planner_rows[instance_id], metric_field)
                if reference_value is None or planner_value is None:
                    continue
                relative_increase = planner_value / reference_value - 1.0
                if abs(relative_increase) < cls.QUALITY_ZERO_TOLERANCE:
                    relative_increase = 0.0
                values.append(relative_increase)
            values_by_planner.append(values)
        return values_by_planner

    @classmethod
    def _figure_y_limits(
        cls,
        values: Sequence[float],
        metric_name: str,
        group: str,
    ) -> tuple[float, float]:
        if metric_name != "quality":
            return super()._figure_y_limits(values, metric_name, group)
        limits = cls.FIGURE_Y_LIMITS[metric_name][group]
        if limits is None:
            return cls._quality_axis_limits(values)
        y_low, y_high = limits
        if y_high <= y_low:
            raise ValueError(
                f"Invalid y-limits for {metric_name}/{group}: "
                f"expected low < high, got {limits}"
            )
        return float(y_low), float(y_high)

    @classmethod
    def _quality_axis_limits(cls, values: Sequence[float]) -> tuple[float, float]:
        finite_values = [value for value in values if math.isfinite(value)]
        if not finite_values:
            return -0.01, cls.QUALITY_Y_AXIS_MIN_HIGH
        low = min(min(finite_values), 0.0)
        high = max(max(finite_values), 0.0)
        if math.isclose(low, high):
            high = max(abs(high), cls.QUALITY_Y_AXIS_MIN_HIGH)
            return -0.08 * high, high
        pad = (high - low) * cls.QUALITY_Y_AXIS_PADDING_FRACTION
        y_low = min(low - pad, -0.08 * max(high, cls.QUALITY_Y_AXIS_MIN_HIGH))
        y_high = max(high + pad, cls.QUALITY_Y_AXIS_MIN_HIGH)
        return y_low, y_high

    @classmethod
    def _plot_boxplot_axis(
        cls,
        ax: plt.Axes,
        values_by_heuristic: Sequence[Sequence[float]],
        metric_name: str,
        group: str,
        y_low: float,
        y_high: float,
        row_idx: int,
        col_idx: int,
    ) -> None:
        if metric_name != "quality":
            super()._plot_boxplot_axis(
                ax,
                values_by_heuristic,
                metric_name,
                group,
                y_low,
                y_high,
                row_idx,
                col_idx,
            )
            cls._set_boxplot_axis_xlabel(ax, metric_name)
            return
        if cls.FIGURE_BOXPLOT_ORIENTATION == "horizontal":
            cls._plot_horizontal_quality_axis(
                ax,
                values_by_heuristic,
                group,
                y_low,
                y_high,
                row_idx,
                col_idx,
            )
            return
        if cls.FIGURE_BOXPLOT_ORIENTATION != "vertical":
            raise ValueError(f"Unsupported boxplot orientation {cls.FIGURE_BOXPLOT_ORIENTATION!r}.")

        positions = tuple(range(1, len(cls.QUALITY_FIGURE_PLANNERS) + 1))
        nonempty = [
            (position, planner, list(values))
            for position, planner, values in zip(
                positions,
                cls.QUALITY_FIGURE_PLANNERS,
                values_by_heuristic,
            )
            if values
        ]
        if nonempty:
            for position, planner, values in nonempty:
                cls._draw_quality_boxplot(
                    ax,
                    position,
                    values,
                    cls.FIGURE_HEURISTIC_COLORS[planner],
                    orientation="vertical",
                )
        else:
            ax.text(
                0.5,
                0.5,
                "no common\nsuccesses",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9.5,
                color=PlotPalette.NEUTRAL_DARK,
            )

        ax.set_xlim(0.5, len(cls.QUALITY_FIGURE_PLANNERS) + 0.5)
        ax.set_ylim(y_low, y_high)
        ax.set_yscale("linear")
        ticks = cls.FIGURE_Y_TICKS[metric_name][group]
        if ticks is None:
            ax.yaxis.set_major_locator(
                MaxNLocator(nbins=4, steps=(1, 2, 2.5, 5, 10), min_n_ticks=3)
            )
        else:
            ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(alpha=0.22, linewidth=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [
                cls._quality_tick_label(planner, values)
                for planner, values in zip(cls.QUALITY_FIGURE_PLANNERS, values_by_heuristic)
            ],
            fontsize=cls.FIGURE_COLUMN_TEXT_SIZE,
        )
        if row_idx == 0:
            ax.set_title(cls.FIGURE_GROUP_LABELS[group], fontsize=cls.FIGURE_COLUMN_TEXT_SIZE, pad=6)
        if col_idx == 0:
            ax.set_ylabel(cls.METRIC_LABELS[metric_name], fontsize=cls.FIGURE_COLUMN_TEXT_SIZE)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=cls.FIGURE_COLUMN_TEXT_SIZE, labelrotation=90)
        ax.tick_params(axis="x", labelsize=cls.FIGURE_COLUMN_TEXT_SIZE, pad=1.5)

    @classmethod
    def _plot_horizontal_quality_axis(
        cls,
        ax: plt.Axes,
        values_by_heuristic: Sequence[Sequence[float]],
        group: str,
        x_low: float,
        x_high: float,
        row_idx: int,
        col_idx: int,
    ) -> None:
        positions = tuple(range(1, len(cls.QUALITY_FIGURE_PLANNERS) + 1))
        nonempty = [
            (position, planner, list(values))
            for position, planner, values in zip(
                positions,
                cls.QUALITY_FIGURE_PLANNERS,
                values_by_heuristic,
            )
            if values
        ]
        if nonempty:
            for position, planner, values in nonempty:
                cls._draw_quality_boxplot(
                    ax,
                    position,
                    values,
                    cls.FIGURE_HEURISTIC_COLORS[planner],
                    orientation="horizontal",
                )
        else:
            ax.text(
                0.5,
                0.5,
                "no common\nsuccesses",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9.5,
                color=PlotPalette.NEUTRAL_DARK,
            )

        ax.set_xlim(x_low, x_high)
        ax.set_ylim(len(cls.QUALITY_FIGURE_PLANNERS) + 0.5, 0.5)
        ax.set_xscale("linear")
        ticks = cls.FIGURE_Y_TICKS["quality"][group]
        if ticks is None:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=(1, 2, 2.5, 5, 10), min_n_ticks=3))
        else:
            ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        cls._draw_heuristic_reference_lines(ax, positions)
        ax.grid(axis="x", alpha=0.22, linewidth=0.7)
        ax.set_yticks(positions)
        ax.set_yticklabels(
            [
                cls._quality_tick_label(planner, values)
                for planner, values in zip(cls.QUALITY_FIGURE_PLANNERS, values_by_heuristic)
            ],
            fontsize=cls.FIGURE_COLUMN_TEXT_SIZE,
        )
        if row_idx == 0:
            ax.set_title(cls.FIGURE_GROUP_LABELS[group], fontsize=cls.FIGURE_COLUMN_TEXT_SIZE, pad=6)
        cls._set_boxplot_axis_xlabel(ax, "quality")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=cls.FIGURE_COLUMN_TEXT_SIZE, pad=1.5)
        ax.tick_params(
            axis="y",
            labelsize=cls.FIGURE_COLUMN_TEXT_SIZE,
            pad=1.5,
            labelleft=col_idx == 0,
        )

    @classmethod
    def _draw_quality_boxplot(
        cls,
        ax: plt.Axes,
        position: int,
        values: Sequence[float],
        color: str,
        orientation: str,
    ) -> None:
        finite_values = [value for value in values if math.isfinite(value)]
        if not finite_values:
            return
        BoxplotStyle.draw(
            ax,
            [finite_values],
            positions=[position],
            widths=cls.FIGURE_BOXPLOT_WIDTH,
            color=color,
            orientation=orientation,
        )
        if cls.FIGURE_SHOW_SCATTER_POINTS:
            BoxplotStyle.scatter_points(
                ax,
                float(position),
                finite_values,
                width=cls.FIGURE_BOXPLOT_WIDTH,
                color=color,
                size=cls.FIGURE_SCATTER_SIZE,
                alpha=cls.FIGURE_SCATTER_ALPHA,
                orientation=orientation,
            )

    @classmethod
    def _set_boxplot_axis_xlabel(cls, ax: plt.Axes, metric_name: str) -> None:
        if cls.FIGURE_BOXPLOT_ORIENTATION == "horizontal":
            ax.set_xlabel(cls.FIGURE_X_LABELS[metric_name], fontsize=cls.FIGURE_COLUMN_TEXT_SIZE)

    @classmethod
    def _quality_tick_label(cls, planner: str, values: Sequence[float]) -> str:
        return cls.PLANNER_SHORT_LABELS[planner]

    @classmethod
    def load_rows_by_planner(
        cls,
        results_root: Path,
        manifest: Path | None = None,
    ) -> Dict[str, list[Dict[str, object]]]:
        rows_by_planner = super().load_rows_by_planner(results_root)
        if manifest is None:
            return rows_by_planner
        instance_ids = {
            record.instance_id
            for record in STHeuristicAblationManifestBuilder.load_manifest(Path(manifest))
        }
        return {
            planner: [
                row
                for row in rows
                if str(row["instance_id"]) in instance_ids
            ]
            for planner, rows in rows_by_planner.items()
        }

    @classmethod
    def plot_pairwise_figure(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        metric: str,
        output_prefix: Path,
        formats: Sequence[str],
    ) -> None:
        del metric
        fig, axes = plt.subplots(
            len(cls.GAP_PLANNERS),
            len(cls.FIGURE_METRIC_ORDER),
            figsize=cls.FIGURE_SIZE,
            squeeze=False,
        )
        for col_idx, metric_name in enumerate(cls.FIGURE_METRIC_ORDER):
            x_values, y_values = cls._gap_axis_values(rows_by_planner, budget, metric_name)
            x_low, x_high = cls._quality_x_axis_limits(x_values) if metric_name == "quality" else cls._axis_limits(x_values)
            y_low, y_high = (
                cls._increase_axis_limits(y_values)
                if metric_name in cls.INCREASE_Y_METRICS
                else cls._gap_axis_limits(y_values)
            )
            for row_idx, planner in enumerate(cls.GAP_PLANNERS):
                ax = axes[row_idx][col_idx]
                cls._plot_gap_panel(
                    ax,
                    rows_by_planner,
                    budget,
                    metric_name,
                    planner,
                    x_low,
                    x_high,
                    y_low,
                    y_high,
                    metric_name == "quality",
                )
                ax.set_ylabel(cls.Y_LABEL_FORMATS[metric_name].format(cls.PLANNER_SHORT_LABELS[planner]), fontsize=11)
                if metric_name == "expanded":
                    ax.yaxis.set_label_coords(*cls.EXPANDED_NODE_GAP_Y_LABEL_COORDS[planner])
                if row_idx == len(cls.GAP_PLANNERS) - 1:
                    ax.set_xlabel(cls.X_LABELS[metric_name], fontsize=10.5)
                else:
                    ax.set_xlabel("")

        legend_handles = [
            cls._legend_handle(style)
            for style in cls.FIGURE_GROUP_STYLES.values()
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=len(legend_handles),
            frameon=False,
            bbox_to_anchor=(0.5, 1.01),
            columnspacing=7,
            fontsize=10.5,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), pad=0.3, w_pad=0.3, h_pad=0.3)
        _save_figure(fig, output_prefix, formats)

    @classmethod
    def _plot_gap_panel(
        cls,
        ax: plt.Axes,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        metric_name: str,
        planner: str,
        x_low: float,
        x_high: float,
        y_low: float,
        y_high: float,
        linear_x: bool,
    ) -> None:
        ax.axhline(0.0, color=PlotPalette.NEUTRAL_MID, linestyle="--", linewidth=1.0)
        has_points = False
        for group, style in cls.FIGURE_GROUP_STYLES.items():
            xs, ys = cls._metric_gap_points(rows_by_planner, group, budget, metric_name, planner)
            if not xs:
                continue
            has_points = True
            ax.scatter(
                xs,
                ys,
                s=28,
                marker=str(style["marker"]),
                facecolors=str(style["facecolor"]),
                edgecolors=str(style["edgecolor"]),
                linewidths=1.15,
                alpha=0.85,
            )
        if not has_points:
            ax.text(
                0.5,
                0.5,
                "no common\nsuccesses",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9.5,
                color=PlotPalette.NEUTRAL_DARK,
            )
        cls._style_gap_axis(
            ax,
            x_low,
            x_high,
            y_low,
            y_high,
            linear_x,
            metric_name in cls.PERCENT_Y_METRICS,
            metric_name == "runtime",
            cls.QUALITY_Y_TICKS if metric_name == "quality" else None,
        )

    @classmethod
    def _gap_axis_values(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        metric_name: str,
    ) -> tuple[list[float], list[float]]:
        x_values: list[float] = []
        y_values: list[float] = []
        for planner in cls.GAP_PLANNERS:
            for group in cls.GROUP_ORDER:
                xs, ys = cls._metric_gap_points(rows_by_planner, group, budget, metric_name, planner)
                x_values.extend(xs)
                y_values.extend(ys)
        return x_values, y_values

    @classmethod
    def _metric_gap_points(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        group: str,
        budget: float,
        metric_name: str,
        planner: str,
    ) -> tuple[list[float], list[float]]:
        metric_field = cls.METRIC_FIELDS[metric_name]
        reference_rows = cls._success_rows_by_instance(
            cls._rows_for_group(rows_by_planner.get(cls.REFERENCE_PLANNER, []), group, budget),
            metric_field,
        )
        planner_rows = cls._success_rows_by_instance(
            cls._rows_for_group(rows_by_planner.get(planner, []), group, budget),
            metric_field,
        )
        xs: list[float] = []
        ys: list[float] = []
        for instance_id in sorted(set(reference_rows) & set(planner_rows)):
            reference_value = cls._finite_metric_value(reference_rows[instance_id], metric_field)
            planner_value = cls._finite_metric_value(planner_rows[instance_id], metric_field)
            if reference_value is None or planner_value is None:
                continue
            xs.append(reference_value)
            ys.append(cls._relative_metric_value(metric_name, reference_value, planner_value))
        return xs, ys

    @staticmethod
    def _relative_metric_value(metric_name: str, reference_value: float, planner_value: float) -> float:
        value = (planner_value - reference_value) / reference_value
        if metric_name == "expanded":
            return -value
        if metric_name == "runtime":
            return abs(value)
        return value

    @staticmethod
    def _gap_axis_limits(values: Sequence[float]) -> tuple[float, float]:
        finite_values = [value for value in values if math.isfinite(value)]
        if not finite_values:
            return -1.0, 1.0
        low = min(min(finite_values), 0.0)
        high = max(max(finite_values), 0.0)
        if math.isclose(low, high):
            pad = max(abs(high) * 0.25, 1.0)
        else:
            pad = max((high - low) * 0.12, 0.1)
        return low - pad, high + pad

    @classmethod
    def _increase_axis_limits(cls, values: Sequence[float]) -> tuple[float, float]:
        finite_values = [value for value in values if math.isfinite(value)]
        if not finite_values:
            return -0.1, 1.0
        high = max(max(finite_values), 0.0)
        if math.isclose(high, 0.0):
            return -0.1, 1.0
        pad = high * cls.INCREASE_AXIS_PADDING_FRACTION
        return -pad, high + pad

    @staticmethod
    def _quality_x_axis_limits(values: Sequence[float]) -> tuple[float, float]:
        finite_values = [value for value in values if math.isfinite(value)]
        if not finite_values:
            return 0.0, 1.0
        low = min(finite_values)
        high = max(finite_values)
        if math.isclose(low, high):
            pad = max(abs(high) * 0.25, 1.0)
        else:
            pad = max((high - low) * 0.08, 0.1)
        return low - pad, high + pad

    @staticmethod
    def _style_gap_axis(
        ax: plt.Axes,
        x_low: float,
        x_high: float,
        y_low: float,
        y_high: float,
        linear_x: bool,
        percent_y: bool,
        prune_lower_y_tick: bool,
        fixed_y_ticks: Sequence[float] | None,
    ) -> None:
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(y_low, y_high)
        ax.set_box_aspect(1.0)
        if linear_x:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        else:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
            ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10, labelOnlyBase=True))
            ax.xaxis.set_minor_formatter(NullFormatter())
        if fixed_y_ticks is None:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="lower" if prune_lower_y_tick else None))
        else:
            ax.yaxis.set_major_locator(FixedLocator(fixed_y_ticks))
        if percent_y:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.grid(alpha=0.22, linewidth=0.7)
        ax.tick_params(axis="both", labelsize=9.5)
        ax.tick_params(axis="y", labelrotation=90, pad=5)
        for label in ax.get_yticklabels():
            label.set_horizontalalignment("center")
            label.set_verticalalignment("center")

    @staticmethod
    def _legend_handle(style: Mapping[str, object]):
        return Line2D(
            [0],
            [0],
            marker=str(style["marker"]),
            color="none",
            label=str(style["label"]),
            markerfacecolor=str(style["facecolor"]),
            markeredgecolor=str(style["edgecolor"]),
            markeredgewidth=1.15,
            markersize=6.5,
        )


class STDominationAblationGroupReportCLI:
    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=(
                "Generate domain-wise runtime, expanded-node, and relative-cost panels "
                "for the ST dominance-check ablation."
            )
        )
        parser.add_argument(
            "--results-root",
            type=Path,
            default=STDominationAblationGroupReport.DEFAULT_RESULTS_ROOT,
        )
        parser.add_argument(
            "--manifest",
            type=Path,
            default=STDominationAblationGroupReport.DEFAULT_MANIFEST,
            help="Filter plotted rows to instance IDs in this manifest.",
        )
        parser.add_argument(
            "--output-prefix",
            type=Path,
            default=STDominationAblationGroupReport.DEFAULT_OUTPUT_PREFIX,
        )
        parser.add_argument("--formats", nargs="+", default=("png", "pdf"))
        parser.add_argument("--budget", type=float, default=None)
        return parser.parse_args()

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        rows_by_planner = STDominationAblationGroupReport.load_rows_by_planner(
            args.results_root,
            args.manifest,
        )
        budget = STDominationAblationGroupReport.select_budget(rows_by_planner, args.budget)
        STDominationAblationGroupReport.plot_boxplot_figure(
            rows_by_planner,
            budget,
            "quality",
            args.output_prefix,
            args.formats,
        )


if __name__ == "__main__":
    STDominationAblationGroupReportCLI.main()
