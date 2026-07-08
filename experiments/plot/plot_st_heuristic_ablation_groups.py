from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import median
from typing import Dict, Mapping, Sequence

from experiments.mpl_paper import configure_matplotlib_for_latex

configure_matplotlib_for_latex(backend="Agg")

from matplotlib.ticker import FixedLocator, FormatStrFormatter, FuncFormatter, NullFormatter

from experiments.st_runners.heuristic_ablation_run_search import STHeuristicAblationRunner
from experiments.plot.plot_results_common import BoxplotStyle, PlotPalette, _save_figure, dedupe_result_rows, plt


class STHeuristicAblationGroupReport:
    DEFAULT_RESULTS_ROOT = Path("data/results/st_planning/heuristic_ablation")
    DEFAULT_OUTPUT_PREFIX = Path("latex/figs/heur_ablation")
    DEFAULT_TABLE_OUTPUT = Path("latex/tables/st_heuristic_ablation_group_table.tex")
    DEFAULT_BUDGET = 600.0

    FIGURE_PLANNERS = (
        "Search(GUB)",
        "Search(h_mot+GUB)",
        "Search(h_tri+GUB)",
        "Search(h_tab+GUB)",
        "Search(h_max+GUB)",
    )
    TABLE_PLANNERS = (
        "Search(h_mot+GUB)",
        "Search(h_tri+GUB)",
        "Search(h_tab+GUB)",
    )
    PLANNER_SHORT_LABELS = {
        "Search(GUB)": r"$h_\text{zero}$",
        "Search(h_mot+GUB)": r"$h_\text{mot}$",
        "Search(h_tri+GUB)": r"$h_\text{tri}$",
        "Search(h_tab+GUB)": r"$h_\text{tab}$",
        "Search(h_max+GUB)": r"$h_\text{max}$",
    }
    TABLE_PAIRS = (
        ("Search(h_mot+GUB)", "Search(h_tri+GUB)"),
        ("Search(h_mot+GUB)", "Search(h_tab+GUB)"),
        ("Search(h_tri+GUB)", "Search(h_tab+GUB)"),
    )
    GROUP_ORDER = ("maze", "grid2d", "iris2d")
    GROUP_LABELS = {
        "maze": "maze",
        "grid2d": "grid2d",
        "iris2d": "Iris/iris2d",
    }
    GROUP_BY_SOURCE_DOMAIN = {
        "maze": "maze",
        "grid2d": "grid2d",
        "iris-2d": "iris2d",
        "iris2d": "iris2d",
    }
    FIGURE_GROUP_ORDER = ("grid2d", "maze", "iris2d")
    FIGURE_GROUP_LABELS = {
        "grid2d": r"rand",
        "maze": r"maze",
        "iris2d": r"iris",
    }
    METRIC_FIELDS = {
        "expanded": "num_expanded_nodes",
        "runtime": "runtime",
    }
    METRIC_LABELS = {
        "expanded": "expanded nodes",
        "runtime": "runtime (s)",
    }
    FIGURE_X_LABELS = METRIC_LABELS
    FIGURE_METRIC_ORDER = ("runtime", "expanded")
    FIGURE_HEURISTIC_COLORS = {
        "Search(GUB)": PlotPalette.heuristic_color("h_zero"),
        "Search(h_mot+GUB)": PlotPalette.heuristic_color("h_mot"),
        "Search(h_tri+GUB)": PlotPalette.heuristic_color("h_tri"),
        "Search(h_tab+GUB)": PlotPalette.heuristic_color("h_tab"),
        "Search(h_max+GUB)": PlotPalette.heuristic_color("h_max"),
    }
    FIGURE_Y_TICKS = {
        "runtime": {
            "grid2d": (1e-1, 1e0, 1e1, 1e2),
            "maze": (1e-1, 1e0, 1e1, 1e2),
            "iris2d": (1e-1, 1e0, 1e1, 1e2),
        },
        "expanded": {
            "grid2d": (1e1, 1e3, 1e5),
            "maze": (1e1, 1e3, 1e5),
            "iris2d": (1e1, 1e3, 1e5),
        },
    }
    # Set a tuple to pin one subplot; None keeps that subplot data-derived.
    FIGURE_Y_LIMITS = {
        "runtime": {
            "grid2d": None,
            "maze": None,
            "iris2d": None,
        },
        "expanded": {
            "grid2d": (3, 3e5),
            "maze": (3, 5e5),
            "iris2d": (3, 3e5),
        },
    }
    FIGURE_Y_SCALES = {
        "runtime": "log",
        "expanded": "log",
    }
    FIGURE_BOXPLOT_WIDTH = BoxplotStyle.DEFAULT_WIDTH
    FIGURE_BOXPLOT_WHIS = BoxplotStyle.DEFAULT_WHIS
    FIGURE_ROW_HEIGHT_RATIOS: tuple[float, ...] | None = None
    FIGURE_ROW_LABELS: tuple[str, ...] | None = None
    FIGURE_ROW_LABEL_POSITIONS: tuple[tuple[float, float], ...] | None = None
    FIGURE_ROW_LABEL_X = -0.12
    FIGURE_ROW_LABEL_Y = 1.03
    FIGURE_SHOW_SCATTER_POINTS = True
    FIGURE_SCATTER_WIDTH = 0.24
    FIGURE_SCATTER_SIZE = 12
    FIGURE_SCATTER_ALPHA = 0.58
    FIGURE_BOXPLOT_ORIENTATION = "horizontal"
    FIGURE_HEURISTIC_REFERENCE_COLOR = PlotPalette.NEUTRAL_MID
    FIGURE_HEURISTIC_REFERENCE_ALPHA = 0.28
    FIGURE_HEURISTIC_REFERENCE_LINEWIDTH = 0.7
    FIGURE_HEURISTIC_REFERENCE_LINESTYLE = "--"
    TABLE_METRIC_LABELS = {
        "expanded": "Median expanded nodes",
        "runtime": "Median runtime (s)",
    }
    FIGURE_SIZE = (6, 4.25)
    FIGURE_COLUMN_TEXT_SIZE = 14

    @classmethod
    def load_rows_by_planner(cls, results_root: Path) -> Dict[str, list[Dict[str, object]]]:
        rows_by_planner: Dict[str, list[Dict[str, object]]] = {}
        for planner in cls.FIGURE_PLANNERS:
            path = STHeuristicAblationRunner.planner_output_path(results_root, planner)
            rows_by_planner[planner] = dedupe_result_rows(STHeuristicAblationRunner.load_result_rows(path))
        return rows_by_planner

    @classmethod
    def select_budget(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        requested_budget: float | None,
    ) -> float:
        if requested_budget is not None:
            return float(requested_budget)
        budgets_by_planner: list[set[float]] = []
        observed_budgets: set[float] = set()
        for planner in cls.FIGURE_PLANNERS:
            planner_budgets: set[float] = set()
            for row in rows_by_planner.get(planner, []):
                budget = float(row["budget"])
                if budget > 0.0:
                    planner_budgets.add(budget)
                    observed_budgets.add(budget)
            if planner_budgets:
                budgets_by_planner.append(planner_budgets)
        if budgets_by_planner:
            common_budgets = set.intersection(*budgets_by_planner)
            if common_budgets:
                return max(common_budgets)
        if observed_budgets:
            return max(observed_budgets)
        return cls.DEFAULT_BUDGET

    @staticmethod
    def _same_budget(row: Dict[str, object], budget: float) -> bool:
        return math.isclose(float(row["budget"]), budget)

    @classmethod
    def _row_group(cls, row: Dict[str, object]) -> str:
        source_domain = str(row.get("source_domain", ""))
        if source_domain in cls.GROUP_BY_SOURCE_DOMAIN:
            return cls.GROUP_BY_SOURCE_DOMAIN[source_domain]
        return str(row["group"])

    @classmethod
    def _rows_for_group(
        cls,
        rows: Sequence[Dict[str, object]],
        group: str,
        budget: float,
    ) -> list[Dict[str, object]]:
        return [
            row
            for row in rows
            if cls._row_group(row) == group and cls._same_budget(row, budget)
        ]

    @staticmethod
    def _finite_metric_value(row: Dict[str, object], metric_field: str) -> float | None:
        value = float(row[metric_field])
        if not math.isfinite(value) or value <= 0.0:
            return None
        return value

    @classmethod
    def _success_rows_by_instance(
        cls,
        rows: Sequence[Dict[str, object]],
        metric_field: str,
    ) -> Dict[str, Dict[str, object]]:
        rows_by_instance: Dict[str, Dict[str, object]] = {}
        for row in rows:
            if not bool(row["is_success"]):
                continue
            if cls._finite_metric_value(row, metric_field) is None:
                continue
            rows_by_instance[str(row["instance_id"])] = row
        return rows_by_instance

    @staticmethod
    def _axis_limits(values: Sequence[float]) -> tuple[float, float]:
        finite_values = [value for value in values if math.isfinite(value) and value > 0.0]
        if not finite_values:
            return 1.0, 10.0
        low = min(finite_values)
        high = max(finite_values)
        if math.isclose(low, high):
            return max(low / 2.0, 1e-3), high * 2.0
        return max(low / 1.6, 1e-3), high * 1.6

    @classmethod
    def _figure_y_limits(
        cls,
        values: Sequence[float],
        metric_name: str,
        group: str,
    ) -> tuple[float, float]:
        limits = cls.FIGURE_Y_LIMITS[metric_name][group]
        if limits is None:
            return cls._axis_limits(values)
        y_low, y_high = limits
        if y_low <= 0.0 or y_high <= y_low:
            raise ValueError(
                f"Invalid y-limits for {metric_name}/{group}: "
                f"expected 0 < low < high, got {limits}"
            )
        return float(y_low), float(y_high)

    @classmethod
    def _metric_values_for_instances(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        planner: str,
        group: str,
        budget: float,
        metric_field: str,
        instance_ids: set[str],
    ) -> list[float]:
        rows = cls._success_rows_by_instance(
            cls._rows_for_group(rows_by_planner.get(planner, []), group, budget),
            metric_field,
        )
        values: list[float] = []
        for instance_id in sorted(instance_ids):
            row = rows.get(instance_id)
            if row is None:
                continue
            value = cls._finite_metric_value(row, metric_field)
            if value is not None:
                values.append(value)
        return values

    @classmethod
    def _boxplot_values(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        group: str,
        budget: float,
        metric_field: str,
    ) -> list[list[float]]:
        common_instance_ids = cls._common_success_instance_ids(
            rows_by_planner,
            group,
            budget,
            metric_field,
            cls.FIGURE_PLANNERS,
        )
        return [
            cls._metric_values_for_instances(
                rows_by_planner,
                planner,
                group,
                budget,
                metric_field,
                common_instance_ids,
            )
            for planner in cls.FIGURE_PLANNERS
        ]

    @classmethod
    def plot_boxplot_figure(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        metric: str,
        output_prefix: Path,
        formats: Sequence[str],
    ) -> None:
        del metric
        values_by_metric_group: dict[tuple[str, str], list[list[float]]] = {}
        y_limits_by_metric_group: dict[tuple[str, str], tuple[float, float]] = {}
        for metric_name in cls.FIGURE_METRIC_ORDER:
            metric_field = cls.METRIC_FIELDS[metric_name]
            for group in cls.FIGURE_GROUP_ORDER:
                values_by_heuristic = cls._boxplot_values(
                    rows_by_planner,
                    group,
                    budget,
                    metric_field,
                )
                values_by_metric_group[(metric_name, group)] = values_by_heuristic
                group_values = [
                    value
                    for values in values_by_heuristic
                    for value in values
                ]
                y_limits_by_metric_group[(metric_name, group)] = cls._figure_y_limits(
                    group_values,
                    metric_name,
                    group,
                )

        fig, axes = plt.subplots(
            len(cls.FIGURE_METRIC_ORDER),
            len(cls.FIGURE_GROUP_ORDER),
            figsize=cls.FIGURE_SIZE,
            sharey=False,
            squeeze=False,
            gridspec_kw=cls._boxplot_gridspec_kwargs(),
        )
        for row_idx, metric_name in enumerate(cls.FIGURE_METRIC_ORDER):
            for col_idx, group in enumerate(cls.FIGURE_GROUP_ORDER):
                cls._plot_boxplot_axis(
                    axes[row_idx][col_idx],
                    values_by_metric_group[(metric_name, group)],
                    metric_name,
                    group,
                    *y_limits_by_metric_group[(metric_name, group)],
                    row_idx,
                    col_idx,
                )
                if col_idx == 0:
                    cls._add_row_label(axes[row_idx][col_idx], row_idx)

        fig.tight_layout(pad=0.35, w_pad=0.35, h_pad=0.35)
        _save_figure(fig, output_prefix, formats)

    @classmethod
    def _add_row_label(cls, ax: plt.Axes, row_idx: int) -> None:
        if cls.FIGURE_ROW_LABELS is None:
            return
        if len(cls.FIGURE_ROW_LABELS) != len(cls.FIGURE_METRIC_ORDER):
            raise ValueError(
                "FIGURE_ROW_LABELS must have one entry per figure row: "
                f"got {cls.FIGURE_ROW_LABELS!r} for {cls.FIGURE_METRIC_ORDER!r}."
            )
        row_label_x, row_label_y = cls._row_label_position(row_idx)
        ax.text(
            row_label_x,
            row_label_y,
            cls.FIGURE_ROW_LABELS[row_idx],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=cls.FIGURE_COLUMN_TEXT_SIZE,
            zorder=10,
            clip_on=False,
        )

    @classmethod
    def _row_label_position(cls, row_idx: int) -> tuple[float, float]:
        if cls.FIGURE_ROW_LABEL_POSITIONS is None:
            return cls.FIGURE_ROW_LABEL_X, cls.FIGURE_ROW_LABEL_Y
        if len(cls.FIGURE_ROW_LABEL_POSITIONS) != len(cls.FIGURE_METRIC_ORDER):
            raise ValueError(
                "FIGURE_ROW_LABEL_POSITIONS must have one (x, y) entry per figure row: "
                f"got {cls.FIGURE_ROW_LABEL_POSITIONS!r} for {cls.FIGURE_METRIC_ORDER!r}."
            )
        return cls.FIGURE_ROW_LABEL_POSITIONS[row_idx]

    @classmethod
    def _boxplot_gridspec_kwargs(cls) -> dict[str, tuple[float, ...]] | None:
        if cls.FIGURE_ROW_HEIGHT_RATIOS is None:
            return None
        if len(cls.FIGURE_ROW_HEIGHT_RATIOS) != len(cls.FIGURE_METRIC_ORDER):
            raise ValueError(
                "FIGURE_ROW_HEIGHT_RATIOS must have one entry per figure row: "
                f"got {cls.FIGURE_ROW_HEIGHT_RATIOS!r} for {cls.FIGURE_METRIC_ORDER!r}."
            )
        return {"height_ratios": cls.FIGURE_ROW_HEIGHT_RATIOS}

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
        if cls.FIGURE_BOXPLOT_ORIENTATION == "horizontal":
            cls._plot_horizontal_boxplot_axis(
                ax,
                values_by_heuristic,
                metric_name,
                group,
                y_low,
                y_high,
                row_idx,
                col_idx,
            )
            return
        if cls.FIGURE_BOXPLOT_ORIENTATION != "vertical":
            raise ValueError(f"Unsupported boxplot orientation {cls.FIGURE_BOXPLOT_ORIENTATION!r}.")

        positions = tuple(range(1, len(cls.FIGURE_PLANNERS) + 1))
        nonempty = [
            (position, planner, list(values))
            for position, planner, values in zip(positions, cls.FIGURE_PLANNERS, values_by_heuristic)
            if values
        ]
        if nonempty:
            for position, planner, values in nonempty:
                BoxplotStyle.draw(
                    ax,
                    [values],
                    positions=[position],
                    widths=cls.FIGURE_BOXPLOT_WIDTH,
                    color=cls.FIGURE_HEURISTIC_COLORS[planner],
                    whis=cls.FIGURE_BOXPLOT_WHIS,
                )
                if cls.FIGURE_SHOW_SCATTER_POINTS:
                    cls._plot_boxplot_scatter_points(
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

        ax.set_xlim(0.35, len(cls.FIGURE_PLANNERS) + 0.65)
        ax.set_ylim(y_low, y_high)
        y_scale = cls.FIGURE_Y_SCALES.get(metric_name, "log")
        if y_scale == "log":
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(cls._format_log_tick))
            ax.yaxis.set_minor_formatter(NullFormatter())
        elif y_scale == "linear":
            ax.set_yscale("linear")
            ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        else:
            raise ValueError(f"Unsupported y-scale for {metric_name}: {y_scale!r}")
        ax.yaxis.set_major_locator(FixedLocator(cls.FIGURE_Y_TICKS[metric_name][group]))
        ax.grid(alpha=0.22, linewidth=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [cls.PLANNER_SHORT_LABELS[planner] for planner in cls.FIGURE_PLANNERS],
            fontsize=cls.FIGURE_COLUMN_TEXT_SIZE,
        )
        if row_idx == 0:
            ax.set_title(cls.FIGURE_GROUP_LABELS[group], fontsize=cls.FIGURE_COLUMN_TEXT_SIZE, pad=5)
        if col_idx == 0:
            ax.set_ylabel(cls.METRIC_LABELS[metric_name], fontsize=cls.FIGURE_COLUMN_TEXT_SIZE)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=cls.FIGURE_COLUMN_TEXT_SIZE, labelrotation=90, pad=1.5)
        ax.tick_params(axis="x", labelsize=cls.FIGURE_COLUMN_TEXT_SIZE, pad=1.5)

    @classmethod
    def _plot_horizontal_boxplot_axis(
        cls,
        ax: plt.Axes,
        values_by_heuristic: Sequence[Sequence[float]],
        metric_name: str,
        group: str,
        x_low: float,
        x_high: float,
        row_idx: int,
        col_idx: int,
    ) -> None:
        positions = tuple(range(1, len(cls.FIGURE_PLANNERS) + 1))
        nonempty = [
            (position, planner, list(values))
            for position, planner, values in zip(positions, cls.FIGURE_PLANNERS, values_by_heuristic)
            if values
        ]
        if nonempty:
            for position, planner, values in nonempty:
                BoxplotStyle.draw(
                    ax,
                    [values],
                    positions=[position],
                    widths=cls.FIGURE_BOXPLOT_WIDTH,
                    color=cls.FIGURE_HEURISTIC_COLORS[planner],
                    whis=cls.FIGURE_BOXPLOT_WHIS,
                    orientation="horizontal",
                )
                if cls.FIGURE_SHOW_SCATTER_POINTS:
                    cls._plot_boxplot_scatter_points(
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
        ax.set_ylim(len(cls.FIGURE_PLANNERS) + 0.65, 0.35)
        x_scale = cls.FIGURE_Y_SCALES.get(metric_name, "log")
        if x_scale == "log":
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(cls._format_log_tick))
            ax.xaxis.set_minor_formatter(NullFormatter())
        elif x_scale == "linear":
            ax.set_xscale("linear")
            ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        else:
            raise ValueError(f"Unsupported x-scale for {metric_name}: {x_scale!r}")
        ax.xaxis.set_major_locator(FixedLocator(cls.FIGURE_Y_TICKS[metric_name][group]))
        cls._draw_heuristic_reference_lines(ax, positions)
        ax.grid(axis="x", alpha=0.22, linewidth=0.7)
        ax.set_yticks(positions)
        ax.set_yticklabels(
            [cls.PLANNER_SHORT_LABELS[planner] for planner in cls.FIGURE_PLANNERS],
            fontsize=cls.FIGURE_COLUMN_TEXT_SIZE,
        )
        if row_idx == 0:
            ax.set_title(cls.FIGURE_GROUP_LABELS[group], fontsize=cls.FIGURE_COLUMN_TEXT_SIZE, pad=5)
        ax.set_xlabel(cls.FIGURE_X_LABELS[metric_name], fontsize=cls.FIGURE_COLUMN_TEXT_SIZE)
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=cls.FIGURE_COLUMN_TEXT_SIZE, pad=1.5)
        ax.tick_params(
            axis="y",
            labelsize=cls.FIGURE_COLUMN_TEXT_SIZE,
            pad=1.5,
            labelleft=col_idx == 0,
        )

    @classmethod
    def _draw_heuristic_reference_lines(cls, ax: plt.Axes, positions: Sequence[int]) -> None:
        for position in positions:
            ax.axhline(
                position,
                color=cls.FIGURE_HEURISTIC_REFERENCE_COLOR,
                linestyle=cls.FIGURE_HEURISTIC_REFERENCE_LINESTYLE,
                linewidth=cls.FIGURE_HEURISTIC_REFERENCE_LINEWIDTH,
                alpha=cls.FIGURE_HEURISTIC_REFERENCE_ALPHA,
                zorder=1,
            )

    @classmethod
    def _plot_boxplot_scatter_points(
        cls,
        ax: plt.Axes,
        position: int,
        values: Sequence[float],
        color: str,
        orientation: str,
    ) -> None:
        jittered_positions = cls._boxplot_scatter_x_positions(position, len(values))
        if orientation == "vertical":
            ax.scatter(
                jittered_positions,
                values,
                s=cls.FIGURE_SCATTER_SIZE,
                marker="o",
                facecolors=color,
                edgecolors="white",
                linewidths=0.25,
                alpha=cls.FIGURE_SCATTER_ALPHA,
                zorder=3,
            )
            return
        if orientation == "horizontal":
            ax.scatter(
                values,
                jittered_positions,
                s=cls.FIGURE_SCATTER_SIZE,
                marker="o",
                facecolors=color,
                edgecolors="white",
                linewidths=0.25,
                alpha=cls.FIGURE_SCATTER_ALPHA,
                zorder=3,
            )
            return
        raise ValueError(f"Unsupported scatter orientation {orientation!r}.")

    @classmethod
    def _boxplot_scatter_x_positions(cls, position: int, count: int) -> list[float]:
        if count <= 1:
            return [float(position)] * count
        span = cls.FIGURE_SCATTER_WIDTH
        return [
            float(position) + span * (idx / (count - 1) - 0.5)
            for idx in range(count)
        ]

    @staticmethod
    def _format_log_tick(value: float, _pos: int) -> str:
        if not math.isfinite(value) or value <= 0.0:
            return ""
        exponent = math.log10(value)
        rounded_exponent = round(exponent)
        if math.isclose(exponent, rounded_exponent, rel_tol=0.0, abs_tol=1e-10):
            return rf"$10^{{{int(rounded_exponent)}}}$"
        if value < 0.01 or value >= 1000.0:
            exponent_floor = math.floor(exponent)
            coefficient = value / (10.0 ** exponent_floor)
            return rf"${coefficient:g}\times10^{{{exponent_floor}}}$"
        return f"{value:g}"

    @classmethod
    def _common_success_instance_ids(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        group: str,
        budget: float,
        metric_field: str,
        planners: Sequence[str],
    ) -> set[str]:
        planner_instance_sets: list[set[str]] = []
        for planner in planners:
            rows = cls._success_rows_by_instance(
                cls._rows_for_group(rows_by_planner.get(planner, []), group, budget),
                metric_field,
            )
            planner_instance_sets.append(set(rows))
        if not planner_instance_sets:
            return set()
        return set.intersection(*planner_instance_sets)

    @staticmethod
    def _format_count(success_count: int, row_count: int) -> str:
        return f"{success_count}/{row_count}" if row_count else "--"

    @staticmethod
    def _format_budget(budget: float) -> str:
        if math.isinf(budget):
            return "no limit"
        if math.isclose(budget, round(budget)):
            return str(int(round(budget)))
        return f"{budget:g}"

    @staticmethod
    def _format_metric_value(value: float | None, metric: str) -> str:
        if value is None or not math.isfinite(value):
            return "--"
        if metric == "expanded":
            return f"{int(round(value)):,}"
        return f"{value:.2f}"

    @staticmethod
    def _format_ratio(value: float | None) -> str:
        if value is None or not math.isfinite(value):
            return "--"
        return f"{value:.2f}$\\times$"

    @classmethod
    def _median_metric_for_instances(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        planner: str,
        group: str,
        budget: float,
        metric_field: str,
        instance_ids: set[str],
    ) -> float | None:
        rows = cls._success_rows_by_instance(
            cls._rows_for_group(rows_by_planner.get(planner, []), group, budget),
            metric_field,
        )
        values = [
            cls._finite_metric_value(rows[instance_id], metric_field)
            for instance_id in sorted(instance_ids)
            if instance_id in rows
        ]
        finite_values = [value for value in values if value is not None]
        return median(finite_values) if finite_values else None

    @classmethod
    def _median_ratio_for_instances(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        numerator_planner: str,
        denominator_planner: str,
        group: str,
        budget: float,
        metric_field: str,
        instance_ids: set[str],
    ) -> float | None:
        numerator_rows = cls._success_rows_by_instance(
            cls._rows_for_group(rows_by_planner.get(numerator_planner, []), group, budget),
            metric_field,
        )
        denominator_rows = cls._success_rows_by_instance(
            cls._rows_for_group(rows_by_planner.get(denominator_planner, []), group, budget),
            metric_field,
        )
        ratios: list[float] = []
        for instance_id in sorted(instance_ids):
            numerator = numerator_rows.get(instance_id)
            denominator = denominator_rows.get(instance_id)
            if numerator is None or denominator is None:
                continue
            numerator_value = cls._finite_metric_value(numerator, metric_field)
            denominator_value = cls._finite_metric_value(denominator, metric_field)
            if numerator_value is None or denominator_value is None:
                continue
            ratios.append(numerator_value / denominator_value)
        return median(ratios) if ratios else None

    @classmethod
    def _success_count(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        planner: str,
        group: str,
        budget: float,
    ) -> tuple[int, int]:
        rows = cls._rows_for_group(rows_by_planner.get(planner, []), group, budget)
        return sum(1 for row in rows if bool(row["is_success"])), len(rows)

    @classmethod
    def _group_instance_count(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        group: str,
        budget: float,
    ) -> int:
        instance_ids = {
            str(row["instance_id"])
            for planner in cls.TABLE_PLANNERS
            for row in cls._rows_for_group(rows_by_planner.get(planner, []), group, budget)
        }
        return len(instance_ids)

    @classmethod
    def build_latex_table(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        metric: str,
    ) -> str:
        metric_field = cls.METRIC_FIELDS[metric]
        budget_label = cls._format_budget(budget)
        budget_phrase = (
            "without a runtime limit"
            if math.isinf(budget)
            else f"at budget {budget_label}s"
        )
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            (
                "\\caption{ST heuristic ablation group summary "
                f"{budget_phrase}. All is the common-success subset used by the "
                "median metrics and ratios. Ratios are per-instance medians; "
                "values above one favor the denominator planner.}"
            ),
            "\\label{tab:st-heuristic-ablation-groups}",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{lccccccccccc}",
            "\\toprule",
            (
                "& & \\multicolumn{3}{c}{Solved} & & "
                f"\\multicolumn{{3}}{{c}}{{{cls.TABLE_METRIC_LABELS[metric]}}} & "
                "\\multicolumn{3}{c}{Median ratio} \\\\"
            ),
            "\\cmidrule(lr){3-5}\\cmidrule(lr){7-9}\\cmidrule(lr){10-12}",
            "Group & Inst. & h_mot & h_tri & h_tab & All & h_mot & h_tri & h_tab & h_mot/h_tri & h_mot/h_tab & h_tri/h_tab \\\\",
            "\\midrule",
        ]
        for group in cls.GROUP_ORDER:
            all_success_ids = cls._common_success_instance_ids(
                rows_by_planner,
                group,
                budget,
                metric_field,
                cls.TABLE_PLANNERS,
            )
            success_cells = [
                cls._format_count(*cls._success_count(rows_by_planner, planner, group, budget))
                for planner in cls.TABLE_PLANNERS
            ]
            median_cells = [
                cls._format_metric_value(
                    cls._median_metric_for_instances(
                        rows_by_planner,
                        planner,
                        group,
                        budget,
                        metric_field,
                        all_success_ids,
                    ),
                    metric,
                )
                for planner in cls.TABLE_PLANNERS
            ]
            ratio_cells = [
                cls._format_ratio(
                    cls._median_ratio_for_instances(
                        rows_by_planner,
                        numerator,
                        denominator,
                        group,
                        budget,
                        metric_field,
                        all_success_ids,
                    )
                )
                for numerator, denominator in cls.TABLE_PAIRS
            ]
            row_cells = [
                cls.GROUP_LABELS[group],
                str(cls._group_instance_count(rows_by_planner, group, budget)),
                *success_cells,
                str(len(all_success_ids)),
                *median_cells,
                *ratio_cells,
            ]
            lines.append(" & ".join(row_cells) + " \\\\")
        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
                "",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def write_latex_table(table_source: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(table_source, encoding="utf-8")
        print(f"Saved LaTeX table to {output_path}")


class STHeuristicAblationGroupReportCLI:
    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Generate domain-wise metric boxplots and a LaTeX table for the ST heuristic ablation."
        )
        parser.add_argument(
            "--results-root",
            type=Path,
            default=STHeuristicAblationGroupReport.DEFAULT_RESULTS_ROOT,
        )
        parser.add_argument(
            "--output-prefix",
            type=Path,
            default=STHeuristicAblationGroupReport.DEFAULT_OUTPUT_PREFIX,
        )
        parser.add_argument(
            "--table-output",
            type=Path,
            default=STHeuristicAblationGroupReport.DEFAULT_TABLE_OUTPUT,
        )
        parser.add_argument("--formats", nargs="+", default=("png", "pdf"))
        parser.add_argument(
            "--metric",
            choices=sorted(STHeuristicAblationGroupReport.METRIC_FIELDS),
            default="expanded",
            help=(
                "Metric used for the LaTeX table; the figure always uses "
                "runtime and expanded-node rows."
            ),
        )
        parser.add_argument("--budget", type=float, default=None)
        return parser.parse_args()

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        rows_by_planner = STHeuristicAblationGroupReport.load_rows_by_planner(args.results_root)
        budget = STHeuristicAblationGroupReport.select_budget(rows_by_planner, args.budget)
        STHeuristicAblationGroupReport.plot_boxplot_figure(
            rows_by_planner,
            budget,
            args.metric,
            args.output_prefix,
            args.formats,
        )
        table_source = STHeuristicAblationGroupReport.build_latex_table(rows_by_planner, budget, args.metric)
        STHeuristicAblationGroupReport.write_latex_table(table_source, args.table_output)


if __name__ == "__main__":
    STHeuristicAblationGroupReportCLI.main()
