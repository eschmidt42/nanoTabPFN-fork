import pandas as pd
import polars as pl
from numpy.typing import NDArray
from plotnine import (
    aes,
    coord_cartesian,
    element_blank,
    element_line,
    facet_wrap,
    geom_line,
    geom_ribbon,
    ggplot,
    guides,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_linetype_manual,
    scale_x_continuous,
    theme,
    theme_minimal,
)


def _summarize_nano_runs(
    nano_runs: list[pl.DataFrame], metric: str
) -> tuple[NDArray, NDArray, NDArray | None]:

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
    values = all_runs.select(run_columns).to_numpy()

    mean = values.mean(axis=1)
    training_times = all_runs["training_time"].to_numpy()
    std = values.std(axis=1, ddof=1) if len(run_columns) > 1 else None

    return training_times, mean, std


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


def _plot_nano_runs_data(
    nano_runs: list[pl.DataFrame],
    metric: str,
    baselines: pl.DataFrame | None,
    baselines_std: pl.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_times, mean, std = _summarize_nano_runs(nano_runs, metric)

    line_data = pd.DataFrame(
        {
            "training_time": training_times,
            "value": mean,
            "series": "nanoTabPFN",
        }
    )
    ribbon_rows = []
    if std is not None:
        ribbon_rows.extend(
            {
                "training_time": time,
                "lower": value - deviation,
                "upper": value + deviation,
                "series": "nanoTabPFN",
            }
            for time, value, deviation in zip(training_times, mean, std)
        )

    if baselines is not None:
        baseline_std_values = {}
        if baselines_std is not None:
            baseline_std_values = dict(baselines_std.select("baseline", metric).iter_rows())

        baseline_rows = []
        end_time = max(training_times)
        for baseline, value in baselines.select("baseline", metric).iter_rows():
            for time in (0.0, end_time):
                baseline_rows.append({"training_time": time, "value": value, "series": baseline})

            deviation = baseline_std_values.get(baseline)
            if deviation is not None:
                ribbon_rows.extend(
                    {
                        "training_time": time,
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
) -> ggplot:
    series = list(dict.fromkeys(line_data["series"]))
    colors = {name: _PLOTNINE_COLORS[i % len(_PLOTNINE_COLORS)] for i, name in enumerate(series)}
    linetypes = {
        name: _PLOTNINE_LINETYPES[i % len(_PLOTNINE_LINETYPES)] for i, name in enumerate(series)
    }

    plot = ggplot(line_data)
    if not ribbon_data.empty:
        plot += geom_ribbon(
            data=ribbon_data,
            mapping=aes(
                x="training_time",
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
            x="training_time",
            y="value",
            color="series",
            linetype="series",
            group="series",
        ),
    )
    plot += scale_color_manual(values=colors, name=None)
    plot += scale_fill_manual(values=colors, name=None)
    plot += scale_linetype_manual(values=linetypes, name=None)
    plot += scale_x_continuous(limits=(0, max_training_time), expand=(0, 0))
    plot += coord_cartesian(ylim=(None, 1))
    plot += labs(
        x="Training time (seconds)" if show_xlabel else None,
        y=metric_label if show_ylabel else None,
    )
    plot += theme_minimal()
    plot += theme(
        legend_position="top" if show_legend else "none",
        panel_grid_major_x=element_blank(),
        panel_grid_major_y=element_line(color="#d9d9d9"),
        axis_ticks_major_y=element_blank(),
        axis_text_x=element_blank() if not show_xtics else None,
        axis_ticks_major_x=element_blank() if not show_xtics else None,
    )
    plot += guides(fill="none", linetype="none")
    return plot


def plot_nano_runs(
    nano_runs: list[pl.DataFrame],
    metric: str = "ROC AUC",
    baselines: pl.DataFrame | None = None,
    baselines_std: pl.DataFrame | None = None,
    show_legend: bool = True,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_xtics: bool = True,
) -> ggplot:
    """Build a plotnine plot of runs and optional baselines from Polars DataFrames."""
    line_data, ribbon_data = _plot_nano_runs_data(
        nano_runs,
        metric,
        baselines,
        baselines_std,
    )
    return _plot_nano_runs(
        line_data,
        ribbon_data,
        metric.split("/")[-1],
        max(line_data["training_time"]),
        show_legend,
        show_xlabel,
        show_ylabel,
        show_xtics,
    )


def plot_run_grid(
    nano_runs: list[pl.DataFrame],
    baselines: pl.DataFrame | None = None,
    baselines_std: pl.DataFrame | None = None,
) -> ggplot:
    """Build a plotnine plot of ROC AUC runs faceted by dataset."""
    datasets = list(dict.fromkeys(col.split("/")[0] for col in nano_runs[0].columns if "/" in col))
    metric = "ROC AUC"

    line_frames = []
    ribbon_frames = []
    for dataset in datasets:
        line_data, ribbon_data = _plot_nano_runs_data(
            nano_runs,
            f"{dataset}/{metric}",
            baselines,
            baselines_std,
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
        max(line_data["training_time"]),
        show_legend=True,
        show_xlabel=True,
        show_ylabel=True,
        show_xtics=True,
    )
    return plot + facet_wrap("dataset", nrow=1)
