from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Dict, Mapping, Sequence

from experiments.mpl_paper import configure_matplotlib_for_latex

configure_matplotlib_for_latex(backend="Agg")

from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from experiments.mrmp_runners.common import MRMPExperiment
from benchmark.planners.mrmp import (
    PBSExpansionAblation,
    WindowedCoordinationAblation,
    WindowedCoordinationSpec,
)
from experiments.plot.plot_results_common import BoxplotStyle, PlotPalette, _save_figure, dedupe_result_rows, plt


class MRMPWindowedCoordinationReport:
    DEFAULT_WINDOWED_RESULTS_ROOT = Path(WindowedCoordinationAblation.DEFAULT_OUTPUT_ROOT)
    DEFAULT_PBS_RESULTS_ROOT = Path(WindowedCoordinationAblation.DEFAULT_REFERENCE_OUTPUT_ROOT)
    DEFAULT_OUTPUT_PREFIX = Path("latex/figs/mrmp_windowed_coordination")
    DOMAINS = ("grid2d", "maze", "iris-2d")
    DOMAIN_LABELS = {
        "grid2d": "rand",
        "maze": "maze",
        "iris-2d": "iris",
    }
    REFERENCE_KEY = "pbs_nc"
    DEFAULT_SPAN_FACTORS = (2.5, 5.0, 10.0)
    FIGURE_SIZE = (5, 2.55)
    SUCCESS_RATE_TICKS = (0.0, 0.5, 1.0)
    SUCCESS_RATE_Y_LIMITS = (-0.05, 1.1)
    X_TICK_LABEL_SIZE = 9.5
    Y_TICK_LABEL_SIZE = 9.5
    Y_TICK_LABEL_PAD = 1.5
    BOXPLOT_OFFSET = 1.16
    BOXPLOT_WIDTH_FACTOR = 0.11
    METRIC_BOXPLOT_BOX_WIDTH = 0.7
    METRIC_SPAN_GROUP_PADDING = 0.85
    SPAN_REFERENCE_COLOR = PlotPalette.NEUTRAL_MID
    SPAN_REFERENCE_ALPHA = 0.28
    SPAN_REFERENCE_LINEWIDTH = 0.7
    SPAN_REFERENCE_LINESTYLE = "--"
    METRIC_SCATTER_SIZE = 7.0
    METRIC_SCATTER_ALPHA = 0.65
    SUCCESS_RUNTIME_X_LIMITS: tuple[float, float] | None = (8, 700)
    SUCCESS_RUNTIME_X_TICKS: tuple[float, ...] | None = (10, 80, 600)
    SOC_X_LIMITS_BY_DOMAIN: dict[str, tuple[float, float] | None] = {
        "grid2d": (80, 600),
        "maze": (220, 5000),
        "iris-2d": (130, 6000),
    }
    SOC_X_TICKS_BY_DOMAIN: dict[str, tuple[float, ...] | None] = {
        "grid2d": (100, 240, 500),
        "maze": (300, 1000, 3000),
        "iris-2d": (200, 1000, 4000),
    }
    MAKESPAN_X_LIMITS_BY_DOMAIN: dict[str, tuple[float, float] | None] = {
        "grid2d": (4, 32),
        "maze": (12, 90),
        "iris-2d": (7, 300),
    }
    MAKESPAN_X_TICKS_BY_DOMAIN: dict[str, tuple[float, ...] | None] = {
        "grid2d": (5, 11, 24),
        "maze": (15, 35, 80),
        "iris-2d": (10, 50, 200),
    }
    BOXPLOT_STYLE_KEYS = ("fixed", "dynamic", "dynamic_beta")
    SUCCESS_RUNTIME_LINESTYLES = ("--", "-", ":")
    SUCCESS_RUNTIME_LABELS = {
        "fixed": "fixed",
        "dynamic": "dynamic",
        "dynamic_beta": "dynamic",
    }
    STYLE_ABBREV_PREFIXES = {
        "fixed": "F",
        "dynamic": "D",
        "dynamic_beta": "H",
    }
    STYLE_BY_KEY = {
        REFERENCE_KEY: {
            "label": r"Full-horizon PBS ($\alpha=\infty$)",
            "color": PlotPalette.MRMP_WINDOW_COLORS[REFERENCE_KEY],
            "linestyle": "None",
            "marker": "o",
        },
        "fixed": {
            "label": "WC (static)",
            "color": PlotPalette.MRMP_WINDOW_COLORS["fixed"],
            "linestyle": "-",
            "marker": "s",
            "markerfacecolor": "none",
        },
        "dynamic": {
            "label": "WC (dynamic)",
            "color": PlotPalette.BLACK,
            "linestyle": "-",
            "marker": "s",
            "markerfacecolor": "none",
        },
        "dynamic_beta": {
            "label": r"WC (dynamic, $\beta=0.5$)",
            "color": PlotPalette.MRMP_WINDOW_COLORS["dynamic_beta"],
            "linestyle": "-",
            "marker": "s",
            "markerfacecolor": "none",
        },
    }

    @classmethod
    def domains_for_arg(cls, domain: str) -> tuple[str, ...]:
        if domain == "all":
            return cls.DOMAINS
        return (domain,)

    @staticmethod
    def _safe_result_name(planner_name: str) -> str:
        return planner_name.replace("/", "_")

    @classmethod
    def reference_result_path(cls, pbs_results_root: str | Path, domain_key: str) -> Path:
        rule = PBSExpansionAblation.default_rule()
        safe_name = cls._safe_result_name(PBSExpansionAblation.planner_name(rule))
        return Path(pbs_results_root) / domain_key / "results" / f"{safe_name}.csv"

    @classmethod
    def windowed_result_path(
        cls,
        windowed_results_root: str | Path,
        domain_key: str,
        spec: WindowedCoordinationSpec,
    ) -> Path:
        safe_name = cls._safe_result_name(WindowedCoordinationAblation.planner_name(spec))
        return Path(windowed_results_root) / domain_key / "results" / f"{safe_name}.csv"

    @classmethod
    def load_rows_by_planner(
        cls,
        pbs_results_root: str | Path,
        windowed_results_root: str | Path,
        domains: Sequence[str],
        specs: Sequence[WindowedCoordinationSpec],
    ) -> Dict[str, list[Dict[str, object]]]:
        rows_by_planner: Dict[str, list[Dict[str, object]]] = {cls.REFERENCE_KEY: []}
        for domain in domains:
            path = cls.reference_result_path(pbs_results_root, domain)
            if path.exists():
                rows_by_planner[cls.REFERENCE_KEY].extend(cls._load_domain_rows(path, domain))
        rows_by_planner[cls.REFERENCE_KEY] = dedupe_result_rows(rows_by_planner[cls.REFERENCE_KEY])

        for spec in specs:
            rows: list[Dict[str, object]] = []
            for domain in domains:
                path = cls.windowed_result_path(windowed_results_root, domain, spec)
                if path.exists():
                    rows.extend(cls._load_domain_rows(path, domain))
            rows_by_planner[spec.key] = dedupe_result_rows(rows)
        return rows_by_planner

    @staticmethod
    def _load_domain_rows(path: str | Path, domain_key: str) -> list[Dict[str, object]]:
        rows = MRMPExperiment.load_result_rows(path)
        for row in rows:
            row["domain_key"] = domain_key
        return rows

    @classmethod
    def row_domain_key(cls, row: Mapping[str, object]) -> str | None:
        domain_key = str(row.get("domain_key", ""))
        if domain_key:
            return domain_key
        instance_id = str(row.get("instance_id", ""))
        for candidate in cls.DOMAINS:
            if f"-{candidate}-" in instance_id:
                return candidate
        return None

    @classmethod
    def rows_for_domain(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        domain_key: str,
    ) -> Dict[str, Sequence[Dict[str, object]]]:
        return {
            planner_key: [
                row for row in rows
                if cls.row_domain_key(row) == domain_key
            ]
            for planner_key, rows in rows_by_planner.items()
        }

    @classmethod
    def rows_for_domains(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        domain_keys: Sequence[str],
    ) -> Dict[str, Sequence[Dict[str, object]]]:
        domain_key_set = set(domain_keys)
        return {
            planner_key: [
                row for row in rows
                if cls.row_domain_key(row) in domain_key_set
            ]
            for planner_key, rows in rows_by_planner.items()
        }

    @classmethod
    def domains_from_rows(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
    ) -> tuple[str, ...]:
        observed_domains = {
            domain_key
            for rows in rows_by_planner.values()
            for row in rows
            if (domain_key := cls.row_domain_key(row)) is not None
        }
        return tuple(domain for domain in cls.DOMAINS if domain in observed_domains)

    @classmethod
    def comparison_specs(
        cls,
        specs: Sequence[WindowedCoordinationSpec],
    ) -> tuple[WindowedCoordinationSpec, ...]:
        comparison_specs = list(specs)
        seen_keys = {spec.key for spec in comparison_specs}
        for spec in specs:
            if spec.mode_key != "dynamic" or not math.isclose(float(spec.execution_horizon_factor), 1.0):
                continue
            for beta in WindowedCoordinationAblation.DEFAULT_DYNAMIC_EXECUTION_HORIZON_FACTORS:
                beta_spec = WindowedCoordinationAblation.spec(
                    "dynamic",
                    spec.window_span_factor,
                    execution_horizon_factor=beta,
                )
                if beta_spec.key in seen_keys:
                    continue
                comparison_specs.append(beta_spec)
                seen_keys.add(beta_spec.key)
        return tuple(comparison_specs)

    @classmethod
    def planner_keys_for_specs(cls, specs: Sequence[WindowedCoordinationSpec]) -> tuple[str, ...]:
        keys = [cls.REFERENCE_KEY]
        seen_keys = {cls.REFERENCE_KEY}
        for spec in specs:
            if spec.key in seen_keys:
                continue
            keys.append(spec.key)
            seen_keys.add(spec.key)
        return tuple(keys)

    @staticmethod
    def rows_for_planner_keys(
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        planner_keys: Sequence[str],
    ) -> Dict[str, Sequence[Dict[str, object]]]:
        return {key: rows_by_planner.get(key, []) for key in planner_keys}

    @staticmethod
    def _positive_budgets(rows: Sequence[Dict[str, object]]) -> set[float]:
        return {
            float(row["budget"])
            for row in rows
            if math.isfinite(float(row["budget"])) and float(row["budget"]) > 0.0
        }

    @classmethod
    def select_budget(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        requested_budget: float | None,
    ) -> float:
        if requested_budget is not None:
            return float(requested_budget)
        budget_sets = [cls._positive_budgets(rows) for rows in rows_by_planner.values()]
        if budget_sets and all(budget_sets):
            common_budgets = set.intersection(*budget_sets)
            if common_budgets:
                return max(common_budgets)
        observed_budgets: set[float] = set()
        for rows in rows_by_planner.values():
            observed_budgets.update(cls._positive_budgets(rows))
        if observed_budgets:
            return max(observed_budgets)
        return 300.0

    @staticmethod
    def _same_budget(row: Dict[str, object], budget: float) -> bool:
        return math.isclose(float(row["budget"]), float(budget))

    @classmethod
    def rows_by_instance(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
    ) -> Dict[str, Dict[str, Dict[str, object]]]:
        rows_by_instance: Dict[str, Dict[str, Dict[str, object]]] = {}
        for planner_key, rows in rows_by_planner.items():
            for row in rows:
                if cls._same_budget(row, budget):
                    rows_by_instance.setdefault(str(row["instance_id"]), {})[planner_key] = row
        return rows_by_instance

    @classmethod
    def common_instance_ids(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
    ) -> set[str]:
        planner_keys = set(rows_by_planner)
        return {
            instance_id
            for instance_id, planner_rows in cls.rows_by_instance(rows_by_planner, budget).items()
            if planner_keys.issubset(planner_rows)
        }

    @classmethod
    def planner_budget_rows(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planner_key: str,
    ) -> list[Dict[str, object]]:
        return [
            row for row in rows_by_planner.get(planner_key, [])
            if cls._same_budget(row, budget)
        ]

    @classmethod
    def budget_row_counts(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
    ) -> Dict[str, int]:
        return {
            planner_key: len(cls.planner_budget_rows(rows_by_planner, budget, planner_key))
            for planner_key in rows_by_planner
        }

    @staticmethod
    def format_budget_row_counts(row_counts: Mapping[str, int]) -> str:
        nonempty_counts = [
            f"{planner_key}={count}"
            for planner_key, count in row_counts.items()
            if count > 0
        ]
        if not nonempty_counts:
            return "none"
        return ", ".join(nonempty_counts)

    @staticmethod
    def _finite_positive(value: object) -> float | None:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            return None
        return numeric

    @classmethod
    def _curve_runtime(cls, row: Dict[str, object], budget: float) -> float | None:
        runtime = cls._finite_positive(row["runtime"])
        if runtime is not None:
            return min(runtime, float(budget)) if math.isfinite(float(budget)) else runtime
        if not bool(row["is_success"]):
            return cls._finite_positive(budget)
        return None

    @classmethod
    def success_rate_points(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planner_key: str,
    ) -> tuple[list[float], list[float]]:
        rows = cls.planner_budget_rows(rows_by_planner, budget, planner_key)
        if not rows:
            return [], []

        events: dict[float, int] = {}
        for row in rows:
            x_value = cls._curve_runtime(row, budget)
            if x_value is None:
                continue
            events.setdefault(x_value, 0)
            if bool(row["is_success"]):
                events[x_value] += 1
        if not events:
            return [], []

        xs: list[float] = []
        ys: list[float] = []
        success_count = 0
        total_count = len(rows)
        for x_value in sorted(events):
            success_count += events[x_value]
            xs.append(x_value)
            ys.append(success_count / total_count)
        return xs, ys

    @classmethod
    def success_rate(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planner_key: str,
    ) -> float | None:
        rows = cls.planner_budget_rows(rows_by_planner, budget, planner_key)
        if not rows:
            return None
        return sum(1 for row in rows if bool(row["is_success"])) / len(rows)

    @classmethod
    def median_field(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planner_key: str,
        field: str,
        success_only: bool = False,
    ) -> float | None:
        values = cls.field_values(rows_by_planner, budget, planner_key, field, success_only=success_only)
        if not values:
            return None
        return statistics.median(values)

    @classmethod
    def field_values(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planner_key: str,
        field: str,
        success_only: bool = False,
    ) -> list[float]:
        rows = cls.planner_budget_rows(rows_by_planner, budget, planner_key)
        values: list[float] = []
        for row in rows:
            if success_only and not bool(row["is_success"]):
                continue
            value = cls._finite_positive(row[field])
            if value is not None:
                values.append(value)
        return values

    @classmethod
    def plot_figure(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        specs: Sequence[WindowedCoordinationSpec],
        budget: float,
        output_prefix: Path,
        formats: Sequence[str],
        domains: Sequence[str] | None = None,
    ) -> None:
        domain_keys = tuple(domains) if domains is not None else cls.domains_from_rows(rows_by_planner)
        if not domain_keys:
            domain_keys = cls.DOMAINS
        span_factors = cls._span_factors(specs)
        comparison_specs = cls.comparison_specs(specs)
        planner_keys = cls.planner_keys_for_specs(comparison_specs)
        comparison_rows_by_planner = cls.rows_for_planner_keys(rows_by_planner, planner_keys)
        merged_rows_by_planner = cls.rows_for_domains(comparison_rows_by_planner, domain_keys)
        fig, axes = plt.subplots(1, 3, figsize=cls.FIGURE_SIZE, squeeze=False)
        panel_axes = axes[0]
        cls._plot_success_by_runtime_panel(
            panel_axes[0],
            merged_rows_by_planner,
            comparison_specs,
            budget,
            span_factors,
            show_labels=True,
            show_xlabel=True,
            show_ylabel=True,
        )
        cls._sync_x_limits(
            (panel_axes[0],),
            x_limits=cls.SUCCESS_RUNTIME_X_LIMITS,
            x_ticks=cls.SUCCESS_RUNTIME_X_TICKS,
        )
        metric_panels = (
            ("sum_of_costs", "SoC", "log", True),
            ("makespan", "makespan", "log", True),
        )
        for ax, (field, xlabel, xscale, show_ylabel) in zip(panel_axes[1:], metric_panels):
            cls._plot_metric_by_span_panel(
                ax,
                merged_rows_by_planner,
                comparison_specs,
                budget,
                span_factors,
                field=field,
                ylabel=xlabel,
                yscale=xscale,
                show_xlabel=True,
                show_ylabel=show_ylabel,
            )

        handles, labels = cls.legend_handles_labels(axes)
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.005),
                ncol=3,
                frameon=False,
                fontsize=8.2,
                borderaxespad=0.2,
                handlelength=1.8,
                handletextpad=0.25,
                labelspacing=0.18,
                columnspacing=2,
            )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.82), pad=0.2, w_pad=0.45)
        _save_figure(fig, output_prefix, formats)

    @classmethod
    def domain_label(cls, domain_key: str) -> str:
        return cls.DOMAIN_LABELS.get(domain_key, domain_key)

    @staticmethod
    def legend_handles_labels(axes) -> tuple[list[object], list[str]]:
        handles_by_label: dict[str, object] = {}
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label and label not in handles_by_label:
                    handles_by_label[label] = handle
        return list(handles_by_label.values()), list(handles_by_label)

    @classmethod
    def _sync_x_limits(
        cls,
        axes: Sequence[plt.Axes],
        x_limits: tuple[float, float] | None = None,
        x_ticks: Sequence[float] | None = None,
    ) -> None:
        if x_limits is not None:
            shared_limits = (float(x_limits[0]), float(x_limits[1]))
            for ax in axes:
                ax.set_xlim(shared_limits)
            cls._set_x_ticks(axes, x_ticks)
            return

        positive_limits: list[tuple[float, float]] = []
        for ax in axes:
            low, high = ax.get_xlim()
            if low > 0.0 and high > low and math.isfinite(low) and math.isfinite(high):
                positive_limits.append((float(low), float(high)))
        if not positive_limits:
            return
        shared_limits = (
            min(low for low, _high in positive_limits),
            max(high for _low, high in positive_limits),
        )
        for ax in axes:
            ax.set_xlim(shared_limits)
        cls._set_x_ticks(axes, x_ticks)

    @staticmethod
    def _set_x_ticks(axes: Sequence[plt.Axes], x_ticks: Sequence[float] | None) -> None:
        if x_ticks is None:
            return
        ticks = [float(tick) for tick in x_ticks]
        for ax in axes:
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{float(value):g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())

    @staticmethod
    def _show_x_tick_labels(axes: Sequence[plt.Axes]) -> None:
        for ax in axes:
            ax.tick_params(axis="x", labelbottom=True)
            for label in ax.get_xticklabels():
                label.set_visible(True)

    @classmethod
    def _apply_domain_x_axes(
        cls,
        axes: Sequence[plt.Axes],
        domain_keys: Sequence[str],
        x_limits_by_domain: Mapping[str, tuple[float, float] | None],
        x_ticks_by_domain: Mapping[str, Sequence[float] | None],
    ) -> None:
        for ax, domain_key in zip(axes, domain_keys):
            x_limits = x_limits_by_domain.get(domain_key)
            if x_limits is not None:
                ax.set_xlim((float(x_limits[0]), float(x_limits[1])))
            cls._set_x_ticks((ax,), x_ticks_by_domain.get(domain_key))

    @classmethod
    def _plot_success_by_span_panel(
        cls,
        ax: plt.Axes,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        specs: Sequence[WindowedCoordinationSpec],
        comparison_specs: Sequence[WindowedCoordinationSpec],
        budget: float,
        span_factors: Sequence[float],
        reference_x: float,
        show_labels: bool,
        show_xlabel: bool = True,
        show_ylabel: bool = True,
    ) -> None:
        reference_value = cls._reference_value(rows_by_planner, budget, "success")
        axis_reference_x = reference_x if reference_value is not None else None
        cls._plot_span_series(
            ax,
            rows_by_planner,
            specs,
            budget,
            field="success",
            show_labels=show_labels,
            reference_x=axis_reference_x,
            reference_value=reference_value,
        )
        cls._plot_success_beta_series(
            ax,
            rows_by_planner,
            comparison_specs,
            budget,
            show_labels=show_labels,
            reference_x=axis_reference_x,
            reference_value=reference_value,
        )
        if axis_reference_x is not None:
            cls._plot_reference_marker(ax, axis_reference_x, reference_value, show_labels)
        cls._style_span_axis(ax, span_factors, axis_reference_x)
        ax.set_xlabel(r"span factor $\alpha$" if show_xlabel else "", fontsize=10)
        ax.set_ylabel("success rate" if show_ylabel else "", fontsize=10)
        cls._style_success_axis(ax, show_tick_labels=show_ylabel)

    @classmethod
    def _plot_metric_by_span_panel(
        cls,
        ax: plt.Axes,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        specs: Sequence[WindowedCoordinationSpec],
        budget: float,
        span_factors: Sequence[float],
        field: str,
        ylabel: str,
        yscale: str | None = None,
        show_xlabel: bool = True,
        show_ylabel: bool = True,
    ) -> None:
        variant_positions, span_reference_positions = cls._metric_variant_positions(specs, span_factors)
        tick_labels_by_position = cls._plot_windowed_boxplots(
            ax,
            rows_by_planner,
            specs,
            budget,
            field=field,
            orientation="horizontal",
            variant_positions=variant_positions,
        )
        cls._style_metric_span_axis(
            ax,
            tuple(variant_positions.values()),
            show_tick_labels=show_ylabel,
            tick_labels_by_position=tick_labels_by_position,
        )
        cls._draw_span_reference_lines(ax, span_reference_positions.values())
        ax.set_xlabel(ylabel, fontsize=10)
        ax.set_ylabel("", fontsize=10)
        if yscale is not None:
            ax.set_xscale(yscale)
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis="x", alpha=0.24, linewidth=0.7)
        ax.tick_params(axis="x", labelsize=cls.Y_TICK_LABEL_SIZE)
        cls._style_y_tick_labels(ax, show_tick_labels=show_ylabel, rotation=0.0)

    @classmethod
    def _plot_success_by_runtime_panel(
        cls,
        ax: plt.Axes,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        specs: Sequence[WindowedCoordinationSpec],
        budget: float,
        span_factors: Sequence[float],
        show_labels: bool,
        show_xlabel: bool = True,
        show_ylabel: bool = True,
    ) -> None:
        runtime_values: list[float] = []
        for spec in sorted(specs, key=cls._success_runtime_sort_key):
            xs, ys = cls.success_rate_curve_points(rows_by_planner, budget, spec.key)
            if not xs:
                continue
            runtime_values.extend(xs)
            style_key = cls._style_key_for_spec(spec)
            style = cls.STYLE_BY_KEY[style_key]
            ax.step(
                xs,
                ys,
                where="post",
                label=cls.success_runtime_label(spec) if show_labels else None,
                color=str(style["color"]),
                linestyle=cls._success_runtime_linestyle(spec, span_factors),
                linewidth=1.2,
            )
        reference_runtime = cls.median_field(rows_by_planner, budget, cls.REFERENCE_KEY, "runtime")
        reference_success = cls.success_rate(rows_by_planner, budget, cls.REFERENCE_KEY)
        if (
            reference_runtime is not None
            and reference_success is not None
            and math.isfinite(float(reference_runtime))
            and float(reference_runtime) > 0.0
        ):
            runtime_values.append(float(reference_runtime))
            cls._plot_reference_marker(ax, float(reference_runtime), float(reference_success), show_labels)
        ax.set_xscale("log")
        if runtime_values:
            ax.set_xlim(cls._positive_log_axis_limits(runtime_values))
        ax.set_xlabel("runtime (s)" if show_xlabel else "", fontsize=10)
        ax.set_ylabel("success rate" if show_ylabel else "", fontsize=10)
        cls._style_success_axis(ax, show_tick_labels=show_ylabel)

    @classmethod
    def success_rate_curve_points(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planner_key: str,
    ) -> tuple[list[float], list[float]]:
        xs, ys = cls.success_rate_points(rows_by_planner, budget, planner_key)
        if not xs:
            return [], []
        if math.isfinite(float(budget)) and float(budget) > 0.0 and not math.isclose(xs[-1], float(budget)):
            xs.append(float(budget))
            ys.append(ys[-1])
        left_anchor = cls._success_runtime_left_anchor(xs)
        if left_anchor < xs[0]:
            xs = [left_anchor, *xs]
            ys = [0.0, *ys]
        return xs, ys

    @staticmethod
    def _success_runtime_left_anchor(xs: Sequence[float]) -> float:
        positive_xs = [float(value) for value in xs if float(value) > 0.0 and math.isfinite(float(value))]
        if not positive_xs:
            return 1e-3
        return max(min(positive_xs) / 2.0, 1e-3)

    @classmethod
    def success_runtime_label(cls, spec: WindowedCoordinationSpec) -> str:
        style_key = cls._style_key_for_spec(spec)
        alpha_label = WindowedCoordinationSpec.factor_label(spec.window_span_factor)
        abbrev = cls._style_abbrev(style_key, spec.window_span_factor)
        if style_key == "dynamic_beta":
            beta_label = WindowedCoordinationSpec.factor_label(spec.execution_horizon_factor)
            return rf"{abbrev}: dynamic, $\alpha={alpha_label}$, $\beta={beta_label}$"
        mode_label = cls.SUCCESS_RUNTIME_LABELS[style_key]
        return rf"{abbrev}: {mode_label}, $\alpha={alpha_label}$"

    @classmethod
    def _style_abbrev(cls, style_key: str, window_span_factor: float) -> str:
        prefix = cls.STYLE_ABBREV_PREFIXES[style_key]
        alpha_label = WindowedCoordinationSpec.factor_label(window_span_factor)
        return f"{prefix}{alpha_label}"

    @classmethod
    def _success_runtime_linestyle(
        cls,
        spec: WindowedCoordinationSpec,
        span_factors: Sequence[float],
    ) -> str:
        sorted_factors = [float(factor) for factor in sorted(set(span_factors))]
        try:
            index = sorted_factors.index(float(spec.window_span_factor))
        except ValueError:
            index = 0
        return cls.SUCCESS_RUNTIME_LINESTYLES[index % len(cls.SUCCESS_RUNTIME_LINESTYLES)]

    @classmethod
    def _success_runtime_sort_key(cls, spec: WindowedCoordinationSpec) -> tuple[int, float]:
        style_order = {"fixed": 0, "dynamic": 1, "dynamic_beta": 2}
        return (style_order[cls._style_key_for_spec(spec)], float(spec.window_span_factor))

    @classmethod
    def _plot_windowed_boxplots(
        cls,
        ax: plt.Axes,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        specs: Sequence[WindowedCoordinationSpec],
        budget: float,
        field: str,
        orientation: str = "vertical",
        variant_positions: Mapping[tuple[float, str], float] | None = None,
    ) -> dict[float, str]:
        position_scales = {
            "fixed": 1.0 / cls.BOXPLOT_OFFSET,
            "dynamic": 1.0,
            "dynamic_beta": cls.BOXPLOT_OFFSET,
        }
        tick_labels_by_position: dict[float, str] = {}
        for style_key in cls.BOXPLOT_STYLE_KEYS:
            position_scale = position_scales[style_key]
            mode_specs = sorted(
                (spec for spec in specs if cls._style_key_for_spec(spec) == style_key),
                key=lambda item: item.window_span_factor,
            )
            values: list[list[float]] = []
            positions: list[float] = []
            widths: list[float] = []
            for spec in mode_specs:
                spec_values = cls.field_values(rows_by_planner, budget, spec.key, field)
                if not spec_values:
                    continue
                if variant_positions is None:
                    position = float(spec.window_span_factor) * position_scale
                    width = position * cls.BOXPLOT_WIDTH_FACTOR
                else:
                    position = variant_positions[(float(spec.window_span_factor), style_key)]
                    width = cls.METRIC_BOXPLOT_BOX_WIDTH
                values.append(spec_values)
                positions.append(position)
                widths.append(width)
                tick_labels_by_position[position] = cls._style_abbrev(style_key, spec.window_span_factor)
            if not values:
                continue
            color = str(cls.STYLE_BY_KEY[style_key]["color"])
            BoxplotStyle.draw(
                ax,
                values,
                positions=positions,
                widths=widths,
                color=color,
                orientation=orientation,
            )
            for position, box_values, width in zip(positions, values, widths):
                BoxplotStyle.scatter_points(
                    ax,
                    float(position),
                    box_values,
                    width=float(width),
                    color=color,
                    size=cls.METRIC_SCATTER_SIZE,
                    alpha=cls.METRIC_SCATTER_ALPHA,
                    orientation=orientation,
                )
        return tick_labels_by_position

    @classmethod
    def _metric_variant_positions(
        cls,
        specs: Sequence[WindowedCoordinationSpec],
        span_factors: Sequence[float],
    ) -> tuple[dict[tuple[float, str], float], dict[float, float]]:
        spec_slots = {
            (float(spec.window_span_factor), cls._style_key_for_spec(spec))
            for spec in specs
        }
        variant_positions: dict[tuple[float, str], float] = {}
        span_reference_positions: dict[float, float] = {}
        position = 0.0
        for factor in sorted({float(factor) for factor in span_factors}):
            positions_for_factor: list[float] = []
            for style_key in cls.BOXPLOT_STYLE_KEYS:
                slot = (factor, style_key)
                if slot not in spec_slots:
                    continue
                variant_positions[slot] = position
                positions_for_factor.append(position)
                position += 1.0
            if positions_for_factor:
                span_reference_positions[factor] = sum(positions_for_factor) / len(positions_for_factor)
        return variant_positions, span_reference_positions

    @classmethod
    def _style_metric_span_axis(
        cls,
        ax: plt.Axes,
        axis_positions: Sequence[float],
        show_tick_labels: bool = True,
        tick_labels_by_position: Mapping[float, str] | None = None,
    ) -> None:
        sorted_axis_positions = sorted(float(position) for position in axis_positions)
        if not sorted_axis_positions:
            return
        if tick_labels_by_position is None:
            ticks = sorted_axis_positions
            labels_by_position = {position: "" for position in ticks}
        elif tick_labels_by_position:
            labels_by_position = dict(tick_labels_by_position)
            ticks = sorted(labels_by_position)
        else:
            ticks = sorted_axis_positions
            labels_by_position = {position: "" for position in ticks}
        ax.set_yscale("linear")
        ax.set_ylim(
            min(ticks) - cls.METRIC_SPAN_GROUP_PADDING,
            max(ticks) + cls.METRIC_SPAN_GROUP_PADDING,
        )
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _position: labels_by_position.get(float(value), ""))
        )
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(
            axis="y",
            labelsize=cls.X_TICK_LABEL_SIZE,
            labelleft=show_tick_labels,
            labelrotation=0.0,
        )

    @classmethod
    def _style_key_for_spec(cls, spec: WindowedCoordinationSpec) -> str:
        if spec.mode_key == "dynamic" and not math.isclose(float(spec.execution_horizon_factor), 1.0):
            return "dynamic_beta"
        return spec.mode_key

    @classmethod
    def _plot_reference_boxplot(
        cls,
        ax: plt.Axes,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        field: str,
        reference_x: float,
        orientation: str = "vertical",
    ) -> None:
        values = cls.field_values(rows_by_planner, budget, cls.REFERENCE_KEY, field)
        if not values:
            return
        color = str(cls.STYLE_BY_KEY[cls.REFERENCE_KEY]["color"])
        BoxplotStyle.draw(
            ax,
            [values],
            positions=[reference_x],
            widths=[reference_x * cls.BOXPLOT_WIDTH_FACTOR],
            color=color,
            orientation=orientation,
        )
        BoxplotStyle.scatter_points(
            ax,
            float(reference_x),
            values,
            width=reference_x * cls.BOXPLOT_WIDTH_FACTOR,
            color=color,
            size=cls.METRIC_SCATTER_SIZE,
            alpha=cls.METRIC_SCATTER_ALPHA,
            orientation=orientation,
        )

    @classmethod
    def _draw_span_reference_lines(cls, ax: plt.Axes, span_positions: Sequence[float]) -> None:
        for position in span_positions:
            ax.axhline(
                float(position),
                color=cls.SPAN_REFERENCE_COLOR,
                linestyle=cls.SPAN_REFERENCE_LINESTYLE,
                linewidth=cls.SPAN_REFERENCE_LINEWIDTH,
                alpha=cls.SPAN_REFERENCE_ALPHA,
                zorder=1,
            )

    @classmethod
    def _plot_span_series(
        cls,
        ax: plt.Axes,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        specs: Sequence[WindowedCoordinationSpec],
        budget: float,
        field: str,
        show_labels: bool = True,
        reference_x: float | None = None,
        reference_value: float | None = None,
    ) -> None:
        modes = ("fixed", "dynamic")
        for mode in modes:
            mode_specs = [spec for spec in specs if spec.mode_key == mode]
            xs: list[float] = []
            ys: list[float] = []
            for spec in sorted(mode_specs, key=lambda item: item.window_span_factor):
                if field != "success":
                    raise ValueError(f"Unsupported span-series field {field!r}.")
                value = cls.success_rate(rows_by_planner, budget, spec.key)
                if value is None:
                    continue
                xs.append(float(spec.window_span_factor))
                ys.append(float(value))
            if not xs:
                continue
            style = cls.STYLE_BY_KEY[mode]
            line_xs = xs
            line_ys = ys
            markevery = None
            if reference_x is not None and reference_value is not None:
                line_xs = [*xs, float(reference_x)]
                line_ys = [*ys, float(reference_value)]
                markevery = list(range(len(xs)))
            ax.plot(
                line_xs,
                line_ys,
                label=str(style["label"]) if show_labels else None,
                color=str(style["color"]),
                linestyle=str(style["linestyle"]),
                marker=str(style["marker"]),
                markerfacecolor=str(style.get("markerfacecolor", style["color"])),
                markeredgecolor=str(style["color"]),
                markeredgewidth=0.9,
                markersize=3.6,
                linewidth=1.25,
                markevery=markevery,
            )

    @classmethod
    def _plot_success_beta_series(
        cls,
        ax: plt.Axes,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        specs: Sequence[WindowedCoordinationSpec],
        budget: float,
        show_labels: bool,
        reference_x: float | None,
        reference_value: float | None,
    ) -> None:
        beta_specs = [
            spec
            for spec in specs
            if spec.mode_key == "dynamic"
            and not math.isclose(float(spec.execution_horizon_factor), 1.0)
        ]
        xs: list[float] = []
        ys: list[float] = []
        for spec in sorted(beta_specs, key=lambda item: item.window_span_factor):
            value = cls.success_rate(rows_by_planner, budget, spec.key)
            if value is None:
                continue
            xs.append(float(spec.window_span_factor))
            ys.append(float(value))
        if not xs:
            return

        style = cls.STYLE_BY_KEY["dynamic_beta"]
        line_xs = xs
        line_ys = ys
        markevery = None
        if reference_value is not None and reference_x is not None:
            line_xs = [*xs, float(reference_x)]
            line_ys = [*ys, float(reference_value)]
            markevery = list(range(len(xs)))
        ax.plot(
            line_xs,
            line_ys,
            label=str(style["label"]) if show_labels else None,
            color=str(style["color"]),
            linestyle=str(style["linestyle"]),
            marker=str(style["marker"]),
            markerfacecolor=str(style.get("markerfacecolor", style["color"])),
            markeredgecolor=str(style["color"]),
            markeredgewidth=0.9,
            markersize=3.6,
            linewidth=1.25,
            markevery=markevery,
        )

    @classmethod
    def _reference_value(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        field: str,
    ) -> float | None:
        if field != "success":
            raise ValueError(f"Unsupported reference-marker field {field!r}.")
        return cls.success_rate(rows_by_planner, budget, cls.REFERENCE_KEY)

    @classmethod
    def _plot_reference_marker(
        cls,
        ax: plt.Axes,
        reference_x: float,
        value: float | None,
        show_label: bool,
    ) -> None:
        if value is None:
            return
        style = cls.STYLE_BY_KEY[cls.REFERENCE_KEY]
        ax.plot(
            [reference_x],
            [float(value)],
            label=str(style["label"]) if show_label else None,
            color=str(style["color"]),
            linestyle=str(style["linestyle"]),
            marker=str(style["marker"]),
            markersize=4.0,
            linewidth=0.0,
            zorder=4,
        )

    @staticmethod
    def _span_factors(specs: Sequence[WindowedCoordinationSpec]) -> list[float]:
        return sorted({float(spec.window_span_factor) for spec in specs})

    @staticmethod
    def _reference_span_x(span_factors: Sequence[float]) -> float:
        positive_factors = [float(factor) for factor in span_factors if float(factor) > 0.0]
        if not positive_factors:
            return 2.0
        return 2.0 * max(positive_factors)

    @classmethod
    def _style_span_axis(
        cls,
        ax: plt.Axes,
        span_factors: Sequence[float],
        reference_x: float | None = None,
        axis: str = "x",
        show_tick_labels: bool = True,
    ) -> None:
        ticks = [float(factor) for factor in span_factors if float(factor) > 0.0]
        if reference_x is not None:
            ticks.append(float(reference_x))
        if axis == "x":
            ax.set_xscale("log", base=2)
            ax.set_xlim(cls._span_axis_limits(ticks))
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda value, position: cls._span_tick_label(value, position, reference_x))
            )
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis="x", labelsize=cls.X_TICK_LABEL_SIZE, labelbottom=show_tick_labels)
            return
        if axis == "y":
            ax.set_yscale("log", base=2)
            ax.set_ylim(cls._span_axis_limits(ticks))
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, position: cls._span_tick_label(value, position, reference_x))
            )
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis="y", labelsize=cls.X_TICK_LABEL_SIZE, labelleft=show_tick_labels)
            return
        raise ValueError(f"Unsupported span axis {axis!r}.")

    @staticmethod
    def _span_axis_limits(factors: Sequence[float]) -> tuple[float, float]:
        return MRMPWindowedCoordinationReport._positive_log_axis_limits(factors)

    @staticmethod
    def _positive_log_axis_limits(values: Sequence[float]) -> tuple[float, float]:
        positive_factors = [float(value) for value in values if float(value) > 0.0 and math.isfinite(float(value))]
        if not positive_factors:
            return (1.0, 2.0)
        low = min(positive_factors)
        high = max(positive_factors)
        if math.isclose(low, high):
            return (low / math.sqrt(2.0), high * math.sqrt(2.0))
        log_low = math.log2(low)
        log_high = math.log2(high)
        padding = 0.08 * (log_high - log_low)
        return (2 ** (log_low - padding), 2 ** (log_high + padding))

    @classmethod
    def _style_for_planner(cls, planner_key: str) -> Dict[str, object]:
        if planner_key == cls.REFERENCE_KEY:
            return cls.STYLE_BY_KEY[cls.REFERENCE_KEY]
        mode = "dynamic" if planner_key.startswith("dynamic_") else "fixed"
        return cls.STYLE_BY_KEY[mode]

    @classmethod
    def _style_success_axis(cls, ax: plt.Axes, show_tick_labels: bool = True) -> None:
        ax.set_ylim(*cls.SUCCESS_RATE_Y_LIMITS)
        ax.yaxis.set_major_locator(FixedLocator(cls.SUCCESS_RATE_TICKS))
        ax.yaxis.set_major_formatter(FuncFormatter(cls._success_rate_tick_label))
        ax.grid(alpha=0.24, linewidth=0.7)
        ax.tick_params(axis="y", labelsize=cls.Y_TICK_LABEL_SIZE, labelleft=show_tick_labels)
        cls._style_y_tick_labels(ax, show_tick_labels=show_tick_labels)

    @classmethod
    def _style_y_tick_labels(
        cls,
        ax: plt.Axes,
        show_tick_labels: bool = True,
        rotation: float = 90.0,
    ) -> None:
        ax.tick_params(
            axis="y",
            labelrotation=rotation,
            pad=cls.Y_TICK_LABEL_PAD,
            labelleft=show_tick_labels,
        )

    @staticmethod
    def _success_rate_tick_label(value: float, _position: int) -> str:
        if math.isclose(float(value), 0.0):
            return "0%"
        return f"{float(value):.0%}"

    @staticmethod
    def _span_tick_label(value: float, _position: int, reference_x: float | None = None) -> str:
        if reference_x is not None and math.isclose(float(value), float(reference_x)):
            return r"$\infty$"
        return f"{float(value):g}"


class MRMPWindowedCoordinationReportCLI:
    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Plot MRMP windowed-coordination ablation against full-horizon PBS."
        )
        parser.add_argument(
            "--windowed-results-root",
            type=Path,
            default=MRMPWindowedCoordinationReport.DEFAULT_WINDOWED_RESULTS_ROOT,
        )
        parser.add_argument(
            "--pbs-results-root",
            type=Path,
            default=MRMPWindowedCoordinationReport.DEFAULT_PBS_RESULTS_ROOT,
        )
        parser.add_argument(
            "--output-prefix",
            type=Path,
            default=MRMPWindowedCoordinationReport.DEFAULT_OUTPUT_PREFIX,
        )
        parser.add_argument("--formats", nargs="+", default=("png", "pdf"))
        parser.add_argument("--budget", type=float, default=None)
        parser.add_argument(
            "--domain",
            choices=("all", *MRMPWindowedCoordinationReport.DOMAINS),
            default="all",
        )
        parser.add_argument(
            "--span-factors",
            type=float,
            nargs="+",
            default=list(MRMPWindowedCoordinationReport.DEFAULT_SPAN_FACTORS),
        )
        parser.add_argument(
            "--modes",
            nargs="+",
            choices=WindowedCoordinationAblation.DEFAULT_MODES,
            default=list(WindowedCoordinationAblation.DEFAULT_MODES),
        )
        return parser.parse_args()

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        domains = MRMPWindowedCoordinationReport.domains_for_arg(args.domain)
        specs = WindowedCoordinationAblation.specs(args.span_factors, args.modes)
        comparison_specs = MRMPWindowedCoordinationReport.comparison_specs(specs)
        rows_by_planner = MRMPWindowedCoordinationReport.load_rows_by_planner(
            args.pbs_results_root,
            args.windowed_results_root,
            domains,
            comparison_specs,
        )
        budget = MRMPWindowedCoordinationReport.select_budget(rows_by_planner, args.budget)
        comparison_rows_by_planner = MRMPWindowedCoordinationReport.rows_for_planner_keys(
            rows_by_planner, MRMPWindowedCoordinationReport.planner_keys_for_specs(comparison_specs)
        )
        row_counts = MRMPWindowedCoordinationReport.budget_row_counts(comparison_rows_by_planner, budget)
        num_rows = sum(row_counts.values())
        if num_rows == 0:
            raise RuntimeError(
                f"No MRMP rows found for full-horizon PBS or windowed variants at budget={budget:g}."
            )
        num_common_instances = len(
            MRMPWindowedCoordinationReport.common_instance_ids(comparison_rows_by_planner, budget)
        )
        count_text = MRMPWindowedCoordinationReport.format_budget_row_counts(row_counts)
        print(
            f"Plotting budget={budget:g} with partial MRMP rows "
            f"(common={num_common_instances}, rows: {count_text})."
        )
        MRMPWindowedCoordinationReport.plot_figure(
            rows_by_planner,
            specs,
            budget,
            args.output_prefix,
            args.formats,
            domains=domains,
        )


if __name__ == "__main__":
    MRMPWindowedCoordinationReportCLI.main()
