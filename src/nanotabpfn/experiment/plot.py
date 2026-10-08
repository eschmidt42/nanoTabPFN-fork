from enum import StrEnum, auto

import pandas as pd
import polars as pl
from numpy.typing import NDArray
from plotnine import (
    aes,
    coord_cartesian,
    element_blank,
    element_line,
    element_text,
    facet_wrap,
    geom_line,
    geom_ribbon,
    ggplot,
    guide_legend,
    guides,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_linetype_manual,
    scale_x_continuous,
    theme,
    theme_minimal,
)


def _align_nano_runs_by_training_time(
    nano_runs: list[pl.DataFrame], metric: str
) -> tuple[pl.DataFrame, list[str]]:
    shared_times = (
        pl.concat([run.select("training_time") for run in nano_runs]).unique().sort("training_time")
    )

    aligned_runs = []

    for i, run in enumerate(nano_runs):
        aligned = (
            shared_times.join(
                run.select("training_time", metric).with_columns(pl.col(metric).fill_nan(None)),
                on="training_time",
                how="left",
            )
            .sort("training_time")
            .with_columns(pl.col(metric).interpolate())
            .rename({metric: f"run_{i}"})
        )
        aligned_runs.append(aligned)

    all_runs = aligned_runs[0]
    for run in aligned_runs[1:]:
        all_runs = all_runs.join(run, on="training_time", how="inner")

    all_runs = all_runs.drop_nulls().sort("training_time")
    run_columns = [f"run_{i}" for i in range(len(aligned_runs))]

    return all_runs, run_columns


def _get_nano_runs_time_summary(
    nano_runs: list[pl.DataFrame], metric: str
) -> tuple[NDArray, NDArray, NDArray | None]:

    all_runs, run_columns = _align_nano_runs_by_training_time(nano_runs, metric)

    values = all_runs.select(run_columns).to_numpy()

    mean = values.mean(axis=1)
    training_times = all_runs["training_time"].to_numpy()
    std = values.std(axis=1, ddof=1) if len(run_columns) > 1 else None

    return training_times, mean, std


def _align_nano_runs_by_iteration(
    nano_runs: list[pl.DataFrame], metric: str
) -> tuple[pl.DataFrame, list[str]]:
    x_col = "iteration"
    shared_iteration = pl.concat([run.select(x_col) for run in nano_runs]).unique().sort(x_col)

    aligned_runs = []

    for i, run in enumerate(nano_runs):
        aligned = (
            shared_iteration.join(
                run.select(x_col, metric).with_columns(pl.col(metric).fill_nan(None)),
                on=x_col,
                how="left",
            )
            .sort(x_col)
            .with_columns(pl.col(metric).interpolate())
            .rename({metric: f"run_{i}"})
        )
        aligned_runs.append(aligned)

    all_runs = aligned_runs[0]
    for run in aligned_runs[1:]:
        all_runs = all_runs.join(run, on=x_col, how="inner")

    all_runs = all_runs.drop_nulls().sort(x_col)
    run_columns = [f"run_{i}" for i in range(len(aligned_runs))]

    return all_runs, run_columns


def _get_nano_runs_iteration_summary(
    nano_runs: list[pl.DataFrame], metric: str
) -> tuple[NDArray, NDArray, NDArray | None]:

    all_runs, run_columns = _align_nano_runs_by_iteration(nano_runs, metric)

    values = all_runs.select(run_columns).to_numpy()

    mean = values.mean(axis=1)
    x_col = "iteration"
    iterations = all_runs[x_col].to_numpy()
    std = values.std(axis=1, ddof=1) if len(run_columns) > 1 else None

    return iterations, mean, std


_PLOTNINE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
]
_PLOTNINE_LINETYPES = [
    "solid",
    "dashed",
    "dashdot",
    "dotted",
]


class XTypesEnum(StrEnum):
    training_time = auto()
    iteration = auto()


def _get_data_to_plot_nano_runs(
    nano_runs: list[pl.DataFrame],
    metric: str,
    baselines: pl.DataFrame | None,
    baselines_std: pl.DataFrame | None,
    x_type: XTypesEnum,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    x_col = str(x_type)

    match x_type:
        case XTypesEnum.training_time:
            x, mean, std = _get_nano_runs_time_summary(nano_runs, metric)
        case XTypesEnum.iteration:
            x, mean, std = _get_nano_runs_iteration_summary(nano_runs, metric)
        case _:
            msg = "x-type option not yet implemented."
            raise NotImplementedError(msg)

    line_data = pd.DataFrame(
        {
            x_col: x,
            "value": mean,
            "series": "nanoTabPFN",
        }
    )
    ribbon_rows = []
    if std is not None:
        ribbon_rows.extend(
            {
                x_col: time,
                "lower": value - deviation,
                "upper": value + deviation,
                "series": "nanoTabPFN",
            }
            for time, value, deviation in zip(x, mean, std)
        )

    if baselines is not None:
        baseline_std_values = {}
        if baselines_std is not None:
            baseline_std_values = dict(baselines_std.select("baseline", metric).iter_rows())

        baseline_rows = []
        end_time = max(x)
        for baseline, value in baselines.select("baseline", metric).iter_rows():
            for time in (0.0, end_time):
                baseline_rows.append({x_col: time, "value": value, "series": baseline})

            deviation = baseline_std_values.get(baseline)
            if deviation is not None:
                ribbon_rows.extend(
                    {
                        x_col: time,
                        "lower": value - deviation,
                        "upper": value + deviation,
                        "series": baseline,
                    }
                    for time in (0.0, end_time)
                )
        line_data = pd.concat([line_data, pd.DataFrame(baseline_rows)], ignore_index=True)

    return line_data, pd.DataFrame(ribbon_rows)


def _plot_nano_runs(
    line_data: pd.DataFrame,
    ribbon_data: pd.DataFrame,
    metric_label: str,
    max_training_time: float,
    show_legend: bool,
    show_xlabel: bool,
    show_ylabel: bool,
    show_xtics: bool,
    x_type: XTypesEnum,
) -> ggplot:

    series = list(dict.fromkeys(line_data["series"]))
    colors = {name: _PLOTNINE_COLORS[i % len(_PLOTNINE_COLORS)] for i, name in enumerate(series)}
    linetypes = {
        name: _PLOTNINE_LINETYPES[i % len(_PLOTNINE_LINETYPES)] for i, name in enumerate(series)
    }
    x_col = str(x_type)

    match x_type:
        case XTypesEnum.training_time:
            xlabel = "Training time (seconds)"
        case XTypesEnum.iteration:
            xlabel = "Training iteration"
        case _:
            msg = "x-type option not yet implemented."
            raise NotImplementedError(msg)

    plot = ggplot(line_data)
    if not ribbon_data.empty:
        plot += geom_ribbon(
            data=ribbon_data,
            mapping=aes(
                x=x_col,
                ymin="lower",
                ymax="upper",
                fill="series",
                group="series",
            ),
            alpha=0.2,
            show_legend=False,
        )
    plot += geom_line(
        mapping=aes(
            x=x_col,
            y="value",
            color="series",
            linetype="series",
            group="series",
        ),
    )
    plot += scale_color_manual(
        values=colors,
        name=None,
    )
    plot += scale_fill_manual(
        values=colors,
        name=None,
    )
    plot += scale_linetype_manual(values=linetypes, name=None)
    plot += scale_x_continuous(limits=(0, max_training_time), expand=(0, 0))
    plot += coord_cartesian(ylim=(None, 1))

    plot += labs(
        x=xlabel if show_xlabel else None,
        y=metric_label if show_ylabel else None,
    )
    plot += theme_minimal()
    plot += theme(
        legend_position="top" if show_legend else "none",
        legend_direction="horizontal",
        legend_text=element_text(size=8),
        panel_grid_major_x=element_blank(),
        panel_grid_major_y=element_line(color="#d9d9d9"),
        axis_ticks_major_y=element_blank(),
        axis_text_x=element_blank() if not show_xtics else None,
        axis_ticks_major_x=element_blank() if not show_xtics else None,
        figure_size=(12, 6),
    )
    plot += guides(
        color=guide_legend(ncol=3, byrow=True),
        linetype=guide_legend(ncol=3, byrow=True),
        fill="none",
    )
    return plot


def plot_single_summary_metric(
    nano_runs: list[pl.DataFrame],
    metric: str = "ROC AUC",
    baselines: pl.DataFrame | None = None,
    baselines_std: pl.DataFrame | None = None,
    show_legend: bool = True,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_xtics: bool = True,
    x_type: XTypesEnum = XTypesEnum.training_time,
) -> ggplot:
    """Build a plotnine plot of runs and optional baselines from Polars DataFrames."""

    line_data, ribbon_data = _get_data_to_plot_nano_runs(
        nano_runs, metric, baselines, baselines_std, x_type
    )
    x_col = str(x_type)

    return _plot_nano_runs(
        line_data,
        ribbon_data,
        metric.split("/")[-1],
        max(line_data[x_col]),
        show_legend,
        show_xlabel,
        show_ylabel,
        show_xtics,
        x_type,
    )


def plot_multiple_summary_metrics(
    nano_runs: list[pl.DataFrame],
    baselines: pl.DataFrame | None = None,
    baselines_std: pl.DataFrame | None = None,
    x_type: XTypesEnum = XTypesEnum.training_time,
) -> ggplot:
    """Build a plotnine plot of ROC AUC runs faceted by dataset."""
    datasets = list(dict.fromkeys(col.split("/")[0] for col in nano_runs[0].columns if "/" in col))
    metric = "ROC AUC"
    x_col = str(x_type)

    line_frames = []
    ribbon_frames = []
    for dataset in datasets:
        line_data, ribbon_data = _get_data_to_plot_nano_runs(
            nano_runs,
            f"{dataset}/{metric}",
            baselines,
            baselines_std,
            x_type,
        )
        line_data["dataset"] = dataset
        ribbon_data["dataset"] = dataset
        line_frames.append(line_data)
        ribbon_frames.append(ribbon_data)

    line_data = pd.concat(line_frames, ignore_index=True)
    ribbon_data = pd.concat(ribbon_frames, ignore_index=True)
    plot = _plot_nano_runs(
        line_data,
        ribbon_data,
        metric,
        max(line_data[x_col]),
        show_legend=True,
        show_xlabel=True,
        show_ylabel=True,
        show_xtics=True,
        x_type=x_type,
    )
    return plot + facet_wrap("dataset", nrow=1)
