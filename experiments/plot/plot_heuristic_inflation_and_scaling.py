from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
from statistics import median
from typing import Mapping, Sequence

from experiments.mpl_paper import configure_matplotlib_for_latex

configure_matplotlib_for_latex(backend="Agg")

from matplotlib.ticker import NullFormatter, NullLocator, PercentFormatter

from experiments.base.heuristic_ablation_run_search import STHeuristicAblationRunner
from experiments.base.heuristic_inflation_run_search import STHeuristicInflationRunCLI
from experiments.base.manifest import BaseBenchmarkRecord, load_manifest
from experiments.base.offline_heuristics import BaseOfflineHeuristicStore
from experiments.mrmp.planner_defs import SearchPlannerSpec
from experiments.plot.plot_results_common import BoxplotStyle, PlotPalette, _save_figure, dedupe_result_rows, plt


class HeuristicInflationAndScalingPlot:
    DEFAULT_INFLATION_RESULTS_ROOT = STHeuristicInflationRunCLI.DEFAULT_OUTPUT_ROOT
    DEFAULT_BASELINE_RESULTS_ROOT = Path("data/st_planning/heuristic_ablation/results")
    DEFAULT_SCALING_MANIFEST = Path("data/stgcs_base/heur_computation_time_scaling/manifest.json")
    DEFAULT_OUTPUT_PREFIX = Path("latex/figs/heur_inflation_and_scaling")
    DEFAULT_BUDGET = STHeuristicInflationRunCLI.DEFAULT_BUDGET
    DEFAULT_FONT_SIZE = 11
    HEURISTICS = STHeuristicInflationRunCLI.DEFAULT_HEURISTICS
    BASELINE_EPSILON = 1.0
    EPSILONS = (1.25, 2.5, 5.0, 10.0)
    COST_BOXPLOT_EPSILONS = EPSILONS
    DOMINATION_STACK = STHeuristicInflationRunCLI.DOMINATION_STACK
    RUNTIME_REDUCTION_Y_TICKS = (0.0, 30.0, 60.0, 90.0)
    RUNTIME_REDUCTION_Y_LOWER_LIMIT = -4.0
    RUNTIME_REDUCTION_Y_UPPER_LIMIT = 104.0
    COST_BOXPLOT_MIN_INCREASE_PERCENT_BY_HEURISTIC = {
        "SC": 0.01,
        "LBG": 0.01,
        "TD": 0.01,
        "Max": 0.01,
    }
    COST_BOXPLOT_WHIS = (5, 95)
    COST_BOXPLOT_WIDTH_FRACTION = 0.09
    COST_BOXPLOT_HORIZONTAL_GROUP_SPACING = 0.58
    COST_BOXPLOT_HORIZONTAL_GROUP_MARGIN = 0.28
    COST_BOXPLOT_HORIZONTAL_HEURISTIC_SPACING = 0.12
    COST_BOXPLOT_SCATTER_WIDTH_FRACTION = 0.55
    COST_BOXPLOT_SCATTER_SIZE = 9
    COST_BOXPLOT_SCATTER_ALPHA = 0.58
    SCALING_HEURISTICS = (
        BaseOfflineHeuristicStore.TD_NAME,
        BaseOfflineHeuristicStore.LBG_NAME,
    )
    SCALING_INLINE_LABEL_TARGET_X = {
        BaseOfflineHeuristicStore.TD_NAME: 125.0,
        BaseOfflineHeuristicStore.LBG_NAME: 130.0,
    }
    SCALING_INLINE_LABEL_Y_FACTOR = {
        BaseOfflineHeuristicStore.TD_NAME: 3.0,
        BaseOfflineHeuristicStore.LBG_NAME: 0.45,
    }
    SCALING_X_LOWER_LIMIT = -10
    SCALING_SIZE_UPPER_LIMIT = 210
    SCALING_X_TICKS = (0, 100, 200)
    HEURISTIC_STYLES = {
        "SC": {
            "label": r"$h_\text{mot}$",
            "color": PlotPalette.heuristic_color("SC"),
            "marker": "o",
        },
        "LBG": {
            "label": r"$h_\text{tri}$",
            "color": PlotPalette.heuristic_color("LBG"),
            "marker": "^",
        },
        "TD": {
            "label": r"$h_\text{tab}$",
            "color": PlotPalette.heuristic_color("TD"),
            "marker": "s",
        },
        "Max": {
            "label": r"$h_\text{max}$",
            "color": PlotPalette.heuristic_color("Max"),
            "marker": "D",
        },
    }
    SCALING_HEURISTIC_STYLES = {
        BaseOfflineHeuristicStore.TD_NAME: HEURISTIC_STYLES["TD"],
        BaseOfflineHeuristicStore.LBG_NAME: HEURISTIC_STYLES["LBG"],
    }

    @classmethod
    def planner_spec(cls, heuristic: str, epsilon: float) -> SearchPlannerSpec:
        return SearchPlannerSpec(str(heuristic), cls.DOMINATION_STACK, epsilon=float(epsilon))

    @classmethod
    def planner_name(cls, heuristic: str, epsilon: float) -> str:
        return cls.planner_spec(heuristic, epsilon).name

    @classmethod
    def method_label(cls, heuristic: str) -> str:
        return str(cls.HEURISTIC_STYLES[str(heuristic)]["label"])

    @classmethod
    def planner_names(cls) -> tuple[str, ...]:
        plotted_names = tuple(
            cls.planner_name(heuristic, epsilon)
            for heuristic in cls.HEURISTICS
            for epsilon in cls.EPSILONS
        )
        baseline_names = tuple(cls.planner_name(heuristic, cls.BASELINE_EPSILON) for heuristic in cls.HEURISTICS)
        return (*baseline_names, *plotted_names)

    @classmethod
    def load_inflation_rows_by_planner(
        cls,
        baseline_results_root: Path,
        inflation_results_root: Path,
    ) -> dict[str, list[dict[str, object]]]:
        rows_by_planner: dict[str, list[dict[str, object]]] = {}
        for heuristic in cls.HEURISTICS:
            baseline_name = cls.planner_name(heuristic, cls.BASELINE_EPSILON)
            baseline_path = STHeuristicAblationRunner.planner_output_path(baseline_results_root, baseline_name)
            rows_by_planner[baseline_name] = dedupe_result_rows(
                STHeuristicAblationRunner.load_result_rows(baseline_path)
            )
            for epsilon in cls.EPSILONS:
                planner_name = cls.planner_name(heuristic, epsilon)
                path = STHeuristicAblationRunner.planner_output_path(inflation_results_root, planner_name)
                rows_by_planner[planner_name] = dedupe_result_rows(STHeuristicAblationRunner.load_result_rows(path))
        return rows_by_planner

    @classmethod
    def select_budget(
        cls,
        rows_by_planner: Mapping[str, Sequence[dict[str, object]]],
        requested_budget: float | None,
    ) -> float:
        if requested_budget is not None:
            return float(requested_budget)
        budgets_by_planner: list[set[float]] = []
        observed_budgets: set[float] = set()
        for planner_name in cls.planner_names():
            planner_budgets: set[float] = set()
            for row in rows_by_planner.get(planner_name, []):
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
    def same_budget(row: dict[str, object], budget: float) -> bool:
        return math.isclose(float(row["budget"]), float(budget))

    @staticmethod
    def finite_positive(value: object) -> float | None:
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            return None
        return number

    @classmethod
    def success_rows_by_instance(
        cls,
        rows: Sequence[dict[str, object]],
        budget: float,
    ) -> dict[str, dict[str, object]]:
        rows_by_instance: dict[str, dict[str, object]] = {}
        for row in rows:
            if not cls.same_budget(row, budget) or not bool(row["is_success"]):
                continue
            rows_by_instance[str(row["instance_id"])] = row
        return rows_by_instance

    @classmethod
    def inflation_median_series(
        cls,
        rows_by_planner: Mapping[str, Sequence[dict[str, object]]],
        heuristic: str,
        budget: float,
        metric: str,
    ) -> tuple[list[float], list[float]]:
        values_by_epsilon = cls.inflation_values_by_epsilon(rows_by_planner, heuristic, budget, metric)
        xs = sorted(values_by_epsilon)
        ys = [median(values_by_epsilon[epsilon]) for epsilon in xs]
        return xs, ys

    @classmethod
    def inflation_values_by_epsilon(
        cls,
        rows_by_planner: Mapping[str, Sequence[dict[str, object]]],
        heuristic: str,
        budget: float,
        metric: str,
    ) -> dict[float, list[float]]:
        baseline_name = cls.planner_name(heuristic, cls.BASELINE_EPSILON)
        baseline_rows = cls.success_rows_by_instance(rows_by_planner.get(baseline_name, []), budget)
        values_by_epsilon: dict[float, list[float]] = {}
        for epsilon in cls.EPSILONS:
            planner_name = cls.planner_name(heuristic, epsilon)
            rows = cls.success_rows_by_instance(rows_by_planner.get(planner_name, []), budget)
            values: list[float] = []
            for instance_id in sorted(set(baseline_rows) & set(rows)):
                base_value = cls.finite_positive(baseline_rows[instance_id][metric])
                value = cls.finite_positive(rows[instance_id][metric])
                if base_value is None or value is None:
                    continue
                if metric == "cost":
                    values.append(100.0 * (value / base_value - 1.0))
                elif metric == "runtime":
                    values.append(100.0 * (1.0 - value / base_value))
                else:
                    raise ValueError(f"Unsupported inflation metric {metric!r}.")
            if values:
                values_by_epsilon[float(epsilon)] = values
        return values_by_epsilon

    @classmethod
    def cost_boxplot_values_by_epsilon(
        cls,
        rows_by_planner: Mapping[str, Sequence[dict[str, object]]],
        heuristic: str,
        budget: float,
    ) -> dict[float, list[float]]:
        values_by_epsilon = cls.inflation_values_by_epsilon(rows_by_planner, heuristic, budget, "cost")
        min_increase = float(cls.COST_BOXPLOT_MIN_INCREASE_PERCENT_BY_HEURISTIC[str(heuristic)])
        filtered_values_by_epsilon: dict[float, list[float]] = {}
        for epsilon, values in values_by_epsilon.items():
            filtered_values = [value for value in values if value > min_increase]
            if filtered_values:
                filtered_values_by_epsilon[float(epsilon)] = filtered_values
        return filtered_values_by_epsilon

    @staticmethod
    def graph_size(record: BaseBenchmarkRecord) -> int:
        return int(record.stgcs_num_vertices) + int(record.stgcs_num_edges)

    @classmethod
    def max_graph_size(cls, manifest_path: Path) -> int | None:
        records = load_manifest(manifest_path)
        if not records:
            return None
        return max(cls.graph_size(record) for record in records)

    @classmethod
    def scaling_size_upper_limit(
        cls,
        manifest_path: Path,
        requested_upper_limit: float | None,
    ) -> float:
        if requested_upper_limit is not None:
            upper_limit = float(requested_upper_limit)
        else:
            max_graph_size = cls.max_graph_size(manifest_path)
            upper_limit = float(max_graph_size) if max_graph_size is not None else float(max(cls.SCALING_X_TICKS))
        if not math.isfinite(upper_limit) or upper_limit <= cls.SCALING_X_LOWER_LIMIT:
            raise ValueError(
                f"Scaling size upper limit must be finite and greater than "
                f"{cls.SCALING_X_LOWER_LIMIT}, got {upper_limit}."
            )
        return upper_limit

    @classmethod
    def timing_points(
        cls,
        manifest_path: Path,
        heuristic: str,
    ) -> list[tuple[int, float, str]]:
        records = load_manifest(manifest_path)
        time_index = BaseOfflineHeuristicStore.load_time_index(manifest_path)
        points: list[tuple[int, float, str]] = []
        for record in records:
            value = time_index.get(record.instance_id, {}).get(heuristic)
            if value is None:
                continue
            runtime = float(value)
            if not math.isfinite(runtime) or runtime <= 0.0:
                continue
            points.append((cls.graph_size(record), runtime, record.instance_id))
        return sorted(points, key=lambda item: (item[0], item[1], item[2]))

    @staticmethod
    def median_curve(points: Sequence[tuple[int, float, str]]) -> tuple[list[int], list[float]]:
        grouped: dict[int, list[float]] = defaultdict(list)
        for graph_size, runtime, _ in points:
            grouped[int(graph_size)].append(float(runtime))
        xs = sorted(grouped)
        ys = [median(grouped[x]) for x in xs]
        return xs, ys

    @staticmethod
    def rotate_y_tick_labels(ax: plt.Axes) -> None:
        ax.tick_params(axis="y", labelrotation=90)
        for label in [*ax.get_yticklabels(), *ax.get_yticklabels(minor=True)]:
            label.set_rotation(90)
            label.set_verticalalignment("center")

    @classmethod
    def draw_epsilon_separators(
        cls,
        ax: plt.Axes,
        epsilons: Sequence[float] | None = None,
        orientation: str = "vertical",
        position_by_epsilon: Mapping[float, float] | None = None,
    ) -> None:
        plotted_epsilons = cls.EPSILONS if epsilons is None else tuple(float(epsilon) for epsilon in epsilons)
        for left, right in zip(plotted_epsilons, plotted_epsilons[1:]):
            left_position = (
                float(position_by_epsilon[float(left)])
                if position_by_epsilon is not None
                else float(left)
            )
            right_position = (
                float(position_by_epsilon[float(right)])
                if position_by_epsilon is not None
                else float(right)
            )
            separator = (
                math.sqrt(left_position * right_position)
                if orientation == "vertical"
                else 0.5 * (left_position + right_position)
            )
            if orientation == "vertical":
                ax.axvline(
                    separator,
                    color=PlotPalette.NEUTRAL_MID,
                    linestyle="--",
                    linewidth=0.75,
                    alpha=0.6,
                    zorder=0.5,
                )
            elif orientation == "horizontal":
                ax.axhline(
                    separator,
                    color=PlotPalette.NEUTRAL_MID,
                    linestyle="--",
                    linewidth=0.75,
                    alpha=0.6,
                    zorder=0.5,
                )
            else:
                raise ValueError(f"Unsupported epsilon separator orientation {orientation!r}.")

    @classmethod
    def plot_inflation_panel(
        cls,
        ax: plt.Axes,
        rows_by_planner: Mapping[str, Sequence[dict[str, object]]],
        budget: float,
        metric: str,
        font_size: float,
    ) -> bool:
        plotted = False
        for heuristic in cls.HEURISTICS:
            xs, ys = cls.inflation_median_series(rows_by_planner, heuristic, budget, metric)
            if not xs:
                continue
            plotted = True
            style = cls.HEURISTIC_STYLES[heuristic]
            ax.plot(
                xs,
                ys,
                color=str(style["color"]),
                marker="s",
                markersize=4.5,
                markerfacecolor="none",
                linewidth=1.6,
                label=cls.method_label(heuristic),
            )
        if metric == "runtime":
            ax.axhline(0.0, color=PlotPalette.NEUTRAL_MID, linestyle="--", linewidth=0.8)
            ax.set_yticks(cls.RUNTIME_REDUCTION_Y_TICKS)
            ax.set_ylim(cls.RUNTIME_REDUCTION_Y_LOWER_LIMIT, cls.RUNTIME_REDUCTION_Y_UPPER_LIMIT)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
            ax.set_ylabel(r"runtime reduction", fontsize=font_size, rotation=90)
        elif metric == "cost":
            ax.axhline(0.0, color=PlotPalette.NEUTRAL_MID, linestyle="--", linewidth=0.8)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
            ax.set_ylabel(r"optimality gap", fontsize=font_size, rotation=90)
        else:
            raise ValueError(f"Unsupported inflation metric {metric!r}.")
        ax.set_xlabel(r"heuristic inflation $\varepsilon$", fontsize=font_size)
        ax.set_xscale("log")
        ax.set_xticks(cls.EPSILONS)
        ax.set_xticklabels([f"{epsilon:g}" for epsilon in cls.EPSILONS], fontsize=font_size)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="both", labelsize=font_size)
        cls.rotate_y_tick_labels(ax)
        ax.grid(alpha=0.25, linewidth=0.7)
        return plotted

    @classmethod
    def plot_cost_boxplot_panel(
        cls,
        ax: plt.Axes,
        rows_by_planner: Mapping[str, Sequence[dict[str, object]]],
        budget: float,
        font_size: float,
        orientation: str = "horizontal",
    ) -> bool:
        return cls.plot_inflation_boxplot_panel(
            ax,
            rows_by_planner,
            budget,
            metric="cost",
            font_size=font_size,
            orientation=orientation,
        )

    @classmethod
    def plot_runtime_boxplot_panel(
        cls,
        ax: plt.Axes,
        rows_by_planner: Mapping[str, Sequence[dict[str, object]]],
        budget: float,
        font_size: float,
    ) -> bool:
        return cls.plot_inflation_boxplot_panel(
            ax,
            rows_by_planner,
            budget,
            metric="runtime",
            font_size=font_size,
        )

    @classmethod
    def plot_inflation_boxplot_panel(
        cls,
        ax: plt.Axes,
        rows_by_planner: Mapping[str, Sequence[dict[str, object]]],
        budget: float,
        metric: str,
        font_size: float,
        orientation: str = "vertical",
    ) -> bool:
        if orientation not in ("vertical", "horizontal"):
            raise ValueError(f"Unsupported boxplot orientation {orientation!r}.")
        plotted = False
        offset_by_heuristic = cls.boxplot_offset_by_heuristic()
        plot_epsilons = cls.inflation_boxplot_epsilons(metric)
        position_by_epsilon = cls.boxplot_position_by_epsilon(plot_epsilons, orientation)
        for heuristic in cls.HEURISTICS:
            values_by_epsilon = cls.inflation_boxplot_values_by_epsilon(
                rows_by_planner,
                heuristic,
                budget,
                metric,
            )
            if not values_by_epsilon:
                continue
            plotted = True
            style = cls.HEURISTIC_STYLES[heuristic]
            positions = [
                cls.boxplot_position(
                    epsilon,
                    heuristic,
                    orientation,
                    position_by_epsilon,
                    offset_by_heuristic,
                )
                for epsilon in plot_epsilons
                if epsilon in values_by_epsilon
            ]
            values = [
                values_by_epsilon[epsilon]
                for epsilon in plot_epsilons
                if epsilon in values_by_epsilon
            ]
            widths = [cls.boxplot_width(position, orientation) for position in positions]
            BoxplotStyle.draw(
                ax,
                values,
                positions=positions,
                widths=widths,
                color=str(style["color"]),
                whis=cls.COST_BOXPLOT_WHIS,
                orientation=orientation,
            )
            for position, sample_values, width in zip(positions, values, widths):
                cls.plot_cost_boxplot_scatter_points(
                    ax,
                    position,
                    sample_values,
                    width,
                    str(style["color"]),
                    orientation=orientation,
                )
            ax.plot(
                [],
                [],
                color=str(style["color"]),
                marker=str(style["marker"]),
                markersize=4.5,
                markerfacecolor="white",
                linewidth=1.6,
                label=cls.method_label(heuristic),
            )
        if metric == "runtime":
            if orientation == "vertical":
                ax.axhline(0.0, color=PlotPalette.NEUTRAL_MID, linestyle="--", linewidth=0.8)
                ax.set_yticks(cls.RUNTIME_REDUCTION_Y_TICKS)
                ax.set_ylim(cls.RUNTIME_REDUCTION_Y_LOWER_LIMIT, cls.RUNTIME_REDUCTION_Y_UPPER_LIMIT)
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
                ax.set_ylabel(r"runtime reduction", fontsize=font_size, rotation=90)
            else:
                ax.axvline(0.0, color=PlotPalette.NEUTRAL_MID, linestyle="--", linewidth=0.8)
                ax.set_xticks(cls.RUNTIME_REDUCTION_Y_TICKS)
                ax.set_xlim(cls.RUNTIME_REDUCTION_Y_LOWER_LIMIT, cls.RUNTIME_REDUCTION_Y_UPPER_LIMIT)
                ax.xaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
                ax.set_xlabel(r"runtime reduction", fontsize=font_size)
        elif metric == "cost":
            if orientation == "vertical":
                ax.axhline(0.0, color=PlotPalette.NEUTRAL_MID, linestyle="--", linewidth=0.8)
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
                ax.set_ylabel(r"optimality gap", fontsize=font_size, rotation=90)
            else:
                ax.axvline(0.0, color=PlotPalette.NEUTRAL_MID, linestyle="--", linewidth=0.8)
                ax.xaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
                ax.set_xlabel(r"optimality gap", fontsize=font_size)
        else:
            raise ValueError(f"Unsupported inflation metric {metric!r}.")
        if orientation == "vertical":
            ax.set_xlabel(r"heuristic inflation $\varepsilon$", fontsize=font_size)
            ax.set_xscale("log")
            ax.set_xticks(plot_epsilons)
            ax.set_xticklabels([f"{epsilon:g}" for epsilon in plot_epsilons], fontsize=font_size)
            cls.draw_epsilon_separators(
                ax,
                plot_epsilons,
                orientation="vertical",
                position_by_epsilon=position_by_epsilon,
            )
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())
            grid_axis = "y"
        else:
            ax.set_ylabel(r"heuristic inflation $\varepsilon$", fontsize=font_size)
            ax.set_yscale("linear")
            y_tick_positions = [position_by_epsilon[float(epsilon)] for epsilon in plot_epsilons]
            ax.set_yticks(y_tick_positions)
            ax.set_yticklabels([f"{epsilon:g}" for epsilon in plot_epsilons], fontsize=font_size)
            ax.set_ylim(
                min(y_tick_positions) - cls.COST_BOXPLOT_HORIZONTAL_GROUP_MARGIN,
                max(y_tick_positions) + cls.COST_BOXPLOT_HORIZONTAL_GROUP_MARGIN,
            )
            cls.draw_epsilon_separators(
                ax,
                plot_epsilons,
                orientation="horizontal",
                position_by_epsilon=position_by_epsilon,
            )
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_formatter(NullFormatter())
            grid_axis = "x"
        ax.tick_params(axis="both", labelsize=font_size)
        cls.rotate_y_tick_labels(ax)
        ax.grid(axis=grid_axis, alpha=0.25, linewidth=0.7)
        return plotted

    @classmethod
    def inflation_boxplot_epsilons(cls, metric: str) -> tuple[float, ...]:
        if metric == "cost":
            return cls.COST_BOXPLOT_EPSILONS
        if metric == "runtime":
            return cls.EPSILONS
        raise ValueError(f"Unsupported inflation metric {metric!r}.")

    @classmethod
    def boxplot_position_by_epsilon(
        cls,
        epsilons: Sequence[float],
        orientation: str,
    ) -> dict[float, float]:
        if orientation == "vertical":
            return {float(epsilon): float(epsilon) for epsilon in epsilons}
        if orientation == "horizontal":
            return {
                float(epsilon): 1.0 + idx * cls.COST_BOXPLOT_HORIZONTAL_GROUP_SPACING
                for idx, epsilon in enumerate(epsilons)
            }
        raise ValueError(f"Unsupported boxplot orientation {orientation!r}.")

    @classmethod
    def boxplot_position(
        cls,
        epsilon: float,
        heuristic: str,
        orientation: str,
        position_by_epsilon: Mapping[float, float],
        offset_by_heuristic: Mapping[str, float],
    ) -> float:
        center = float(position_by_epsilon[float(epsilon)])
        if orientation == "vertical":
            return center * float(offset_by_heuristic[heuristic])
        if orientation == "horizontal":
            return center + cls.horizontal_boxplot_heuristic_offset(heuristic)
        raise ValueError(f"Unsupported boxplot orientation {orientation!r}.")

    @classmethod
    def horizontal_boxplot_heuristic_offset(cls, heuristic: str) -> float:
        heuristic_idx = tuple(cls.HEURISTICS).index(str(heuristic))
        midpoint = 0.5 * (len(cls.HEURISTICS) - 1)
        return (float(heuristic_idx) - midpoint) * cls.COST_BOXPLOT_HORIZONTAL_HEURISTIC_SPACING

    @classmethod
    def boxplot_width(cls, position: float, orientation: str) -> float:
        if orientation == "vertical":
            return float(position) * cls.COST_BOXPLOT_WIDTH_FRACTION
        if orientation == "horizontal":
            return cls.COST_BOXPLOT_WIDTH_FRACTION
        raise ValueError(f"Unsupported boxplot orientation {orientation!r}.")

    @classmethod
    def inflation_boxplot_values_by_epsilon(
        cls,
        rows_by_planner: Mapping[str, Sequence[dict[str, object]]],
        heuristic: str,
        budget: float,
        metric: str,
    ) -> dict[float, list[float]]:
        if metric == "cost":
            return cls.cost_boxplot_values_by_epsilon(rows_by_planner, heuristic, budget)
        if metric == "runtime":
            return cls.inflation_values_by_epsilon(rows_by_planner, heuristic, budget, metric)
        raise ValueError(f"Unsupported inflation metric {metric!r}.")

    @staticmethod
    def boxplot_offset_by_heuristic() -> dict[str, float]:
        return {
            "SC": 0.79,
            "LBG": 0.93,
            "TD": 1.07,
            "Max": 1.21,
        }

    @classmethod
    def plot_cost_boxplot_scatter_points(
        cls,
        ax: plt.Axes,
        position: float,
        values: Sequence[float],
        width: float,
        color: str,
        orientation: str = "vertical",
    ) -> None:
        xs = cls.cost_boxplot_scatter_x_positions(position, width, len(values))
        if orientation == "vertical":
            ax.scatter(
                xs,
                values,
                s=cls.COST_BOXPLOT_SCATTER_SIZE,
                marker="o",
                facecolors=color,
                edgecolors="white",
                linewidths=0.25,
                alpha=cls.COST_BOXPLOT_SCATTER_ALPHA,
                zorder=3,
            )
            return
        if orientation == "horizontal":
            ax.scatter(
                values,
                xs,
                s=cls.COST_BOXPLOT_SCATTER_SIZE,
                marker="o",
                facecolors=color,
                edgecolors="white",
                linewidths=0.25,
                alpha=cls.COST_BOXPLOT_SCATTER_ALPHA,
                zorder=3,
            )
            return
        raise ValueError(f"Unsupported boxplot scatter orientation {orientation!r}.")

    @classmethod
    def cost_boxplot_scatter_x_positions(cls, position: float, width: float, count: int) -> list[float]:
        if count <= 1:
            return [float(position)] * count
        span = float(width) * cls.COST_BOXPLOT_SCATTER_WIDTH_FRACTION
        return [
            float(position) + span * (idx / (count - 1) - 0.5)
            for idx in range(count)
        ]

    @classmethod
    def plot_scaling_panel(
        cls,
        ax: plt.Axes,
        manifest_path: Path,
        font_size: float,
        size_upper_limit: float | None = SCALING_SIZE_UPPER_LIMIT,
    ) -> bool:
        plotted = False
        for heuristic in cls.SCALING_HEURISTICS:
            points = cls.timing_points(manifest_path, heuristic)
            if not points:
                print(f"No {heuristic} timing records found for {manifest_path}")
                continue
            plotted = True
            style = cls.SCALING_HEURISTIC_STYLES[heuristic]
            curve_xs, curve_ys = cls.median_curve(points)
            ax.plot(
                curve_xs,
                curve_ys,
                color=str(style["color"]),
                linewidth=1.6,
                label=cls.method_label(str(heuristic)),
            )
            cls.draw_scaling_inline_label(ax, heuristic, curve_xs, curve_ys, font_size)
        ax.set_yscale("log")
        ax.set_yticks((1e-5, 1e-1, 1e3))
        ax.set_yticklabels((r"$10^{-5}$", r"$10^{-1}$", r"$10^{3}$"), fontsize=font_size)
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"$G_0$ size ($|V_0| + |E_0|$)", fontsize=font_size)
        x_upper_limit = cls.scaling_size_upper_limit(manifest_path, size_upper_limit)
        ax.set_xticks(tuple(tick for tick in cls.SCALING_X_TICKS if tick <= x_upper_limit))
        ax.set_xlim(cls.SCALING_X_LOWER_LIMIT, x_upper_limit)
        if ax.get_xticklabels():
            ax.get_xticklabels()[-1].set_horizontalalignment("center")
        ax.set_ylabel("precomputation time (s)", fontsize=font_size, rotation=90)
        ax.tick_params(axis="both", labelsize=font_size)
        ax.tick_params(axis="y", which="both", left=True, labelleft=True)
        cls.rotate_y_tick_labels(ax)
        ax.grid(alpha=0.25, linewidth=0.7)
        return plotted

    @classmethod
    def draw_scaling_inline_label(
        cls,
        ax: plt.Axes,
        heuristic: str,
        curve_xs: Sequence[int],
        curve_ys: Sequence[float],
        font_size: float,
    ) -> None:
        if not curve_xs or not curve_ys:
            return
        target_x = float(cls.SCALING_INLINE_LABEL_TARGET_X[str(heuristic)])
        label_idx = min(range(len(curve_xs)), key=lambda idx: abs(float(curve_xs[idx]) - target_x))
        style = cls.SCALING_HEURISTIC_STYLES[str(heuristic)]
        y_factor = float(cls.SCALING_INLINE_LABEL_Y_FACTOR[str(heuristic)]) / 25
        ax.text(
            float(curve_xs[label_idx]) - 100,
            float(curve_ys[label_idx]) * y_factor,
            cls.method_label(str(heuristic)),
            color=str(style["color"]),
            fontsize=font_size,
            ha="left",
            va="center",
            zorder=4,
        )

    @classmethod
    def plot(
        cls,
        baseline_results_root: Path,
        inflation_results_root: Path,
        scaling_manifest_path: Path,
        output_prefix: Path,
        formats: Sequence[str],
        budget: float | None = None,
        font_size: float = DEFAULT_FONT_SIZE,
        scaling_size_upper_limit: float | None = SCALING_SIZE_UPPER_LIMIT,
    ) -> None:
        font_size = float(font_size)
        if not math.isfinite(font_size) or font_size <= 0.0:
            raise ValueError(f"font_size must be positive and finite, got {font_size}.")

        rows_by_planner = cls.load_inflation_rows_by_planner(
            baseline_results_root=baseline_results_root,
            inflation_results_root=inflation_results_root,
        )
        selected_budget = cls.select_budget(rows_by_planner, budget)
        fig = plt.figure(figsize=(4.5, 4))
        grid_spec = fig.add_gridspec(
            2,
            2,
            width_ratios=(1.0, 1.45),
            height_ratios=(1.0, 1.0),
            wspace=0.32,
            hspace=0.45,
        )
        runtime_ax = fig.add_subplot(grid_spec[0, 0])
        scaling_ax = fig.add_subplot(grid_spec[1, 0])
        cost_ax = fig.add_subplot(grid_spec[:, 1])
        runtime_plotted = cls.plot_inflation_panel(
            runtime_ax,
            rows_by_planner,
            selected_budget,
            metric="runtime",
            font_size=font_size,
        )
        cost_plotted = cls.plot_cost_boxplot_panel(
            cost_ax,
            rows_by_planner,
            selected_budget,
            font_size=font_size,
            orientation="horizontal",
        )
        scaling_plotted = cls.plot_scaling_panel(
            scaling_ax,
            scaling_manifest_path,
            font_size,
            scaling_size_upper_limit,
        )
        if not runtime_plotted or not cost_plotted:
            raise ValueError(
                f"No heuristic-inflation result series found in {baseline_results_root} "
                f"and {inflation_results_root}."
            )
        if not scaling_plotted:
            raise ValueError(f"No TD/LBG timing records found for {scaling_manifest_path}.")

        handles, labels = runtime_ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(labels),
            frameon=False,
            bbox_to_anchor=(0.5, 0.95),
            fontsize=font_size,
            columnspacing=2,
        )
        fig.subplots_adjust(left=0.1, right=0.98, bottom=0.12, top=0.86)
        _save_figure(fig, output_prefix, formats)


class HeuristicInflationAndScalingPlotCLI:
    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=(
                "Plot BFS heuristic-inflation effects under GUB-only domination and offline TD/LBG "
                "heuristic computation time."
            )
        )
        parser.add_argument(
            "--baseline-results-root",
            type=Path,
            default=HeuristicInflationAndScalingPlot.DEFAULT_BASELINE_RESULTS_ROOT,
        )
        parser.add_argument(
            "--inflation-results-root",
            type=Path,
            default=HeuristicInflationAndScalingPlot.DEFAULT_INFLATION_RESULTS_ROOT,
        )
        parser.add_argument(
            "--scaling-manifest",
            type=Path,
            default=HeuristicInflationAndScalingPlot.DEFAULT_SCALING_MANIFEST,
        )
        parser.add_argument("--output-prefix", type=Path, default=HeuristicInflationAndScalingPlot.DEFAULT_OUTPUT_PREFIX)
        parser.add_argument("--formats", nargs="+", default=("pdf", "png"))
        parser.add_argument("--budget", type=float, default=None)
        parser.add_argument("--font-size", type=float, default=HeuristicInflationAndScalingPlot.DEFAULT_FONT_SIZE)
        parser.add_argument(
            "--scaling-size-upper-limit",
            "--scaling-x-upper-limit",
            type=float,
            default=HeuristicInflationAndScalingPlot.SCALING_SIZE_UPPER_LIMIT,
            help="Upper limit for the ST-GCS size axis; defaults to the max size in the scaling manifest.",
        )
        return parser.parse_args()

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        HeuristicInflationAndScalingPlot.plot(
            baseline_results_root=args.baseline_results_root,
            inflation_results_root=args.inflation_results_root,
            scaling_manifest_path=args.scaling_manifest,
            output_prefix=args.output_prefix,
            formats=tuple(str(fmt) for fmt in args.formats),
            budget=None if args.budget is None else float(args.budget),
            font_size=float(args.font_size),
            scaling_size_upper_limit=(
                None if args.scaling_size_upper_limit is None else float(args.scaling_size_upper_limit)
            ),
        )


if __name__ == "__main__":
    HeuristicInflationAndScalingPlotCLI.main()
