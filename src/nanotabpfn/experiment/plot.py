import functools

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns


def plot_nano_runs(
    ax: plt.Axes,
    nano_runs: list[pl.DataFrame],
    metric: str,
    baselines: pl.DataFrame | None = None,
    baselines_std: pl.DataFrame | None = None,
    show_legend: bool = True,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_xtics: bool = True,
):
    """
    Plot runs and optional baselines using Polars DataFrames.

    Each run needs `"training_time"` and `metric` columns. Baseline frames need
    a `"baseline"` column containing the label for each row and a `metric` column.
    """
    colors = sns.color_palette("tab10")[1:]
    linestyles = [
        "--",
        "-.",
        ":",
        (0, (3, 1, 1, 1)),
        (0, (5, 5)),
    ]

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
    ax.plot(training_times, mean, label="nanoTabPFN", zorder=2, color="blue")
    if len(run_columns) > 1:
        std = values.std(axis=1, ddof=1)
        ax.fill_between(training_times, mean - std, mean + std, alpha=0.2, zorder=2)

    if baselines is not None:
        baseline_std_values = (
            dict(baselines_std.select("baseline", metric).iter_rows())
            if baselines_std is not None
            else {}
        )
        for i, (baseline_name, baseline_value) in enumerate(
            baselines.select("baseline", metric).iter_rows()
        ):
            color = colors[i % len(colors)]
            ax.plot(
                [0, max(training_times)],
                [baseline_value, baseline_value],
                label=baseline_name,
                alpha=0.7,
                linestyle=linestyles[i],
                color=color,
                zorder=1,
            )
            if baseline_name in baseline_std_values:
                std = baseline_std_values[baseline_name]
                ax.fill_between(
                    [0, max(training_times)],
                    [baseline_value - std, baseline_value - std],
                    [baseline_value + std, baseline_value + std],
                    alpha=0.2,
                    zorder=1,
                )

    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.tick_params(axis="y", length=0)
    if not show_xtics:
        ax.tick_params(axis="x", length=0)
    if show_xlabel:
        ax.set_xlabel("Training time (seconds)")
    if show_ylabel:
        ax.set_ylabel(metric.split("/")[-1])
    max_time = max(training_times)
    ax.set_xlim(0, max_time)
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], 1)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        label_y_values = {}
        for handle, label in zip(handles, labels):
            if isinstance(handle, plt.Line2D):
                y_data = np.asarray(handle.get_ydata())
                label_y_values[label] = y_data[-1]
        sorted_labels = sorted(label_y_values.items(), key=lambda x: x[1], reverse=True)
        sorted_handles = [
            handle
            for label, _ in sorted_labels
            for handle, lbl in zip(handles, labels)
            if lbl == label
        ]
        sorted_labels = [label for label, _ in sorted_labels]
        ax.legend(sorted_handles, sorted_labels)

    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_run_grid(
    nano_runs: list[pl.DataFrame],
    baselines: pl.DataFrame | None = None,
    baselines_std: pl.DataFrame | None = None,
):
    """Plot Polars runs in a grid of metrics x datasets."""
    datasets = list(dict.fromkeys(col.split("/")[0] for col in nano_runs[0].columns if "/" in col))
    metric = "ROC AUC"
    figsize = (len(datasets) * 4, 4.6)
    fig, axs = plt.subplots(
        1, len(datasets), figsize=figsize, sharex=True, sharey=True, layout="constrained"
    )
    axs = np.atleast_1d(axs)
    fig.set_layout_engine("constrained", w_pad=0.0, h_pad=0.1)
    for j, dataset in enumerate(datasets):
        ax = axs[j]
        plot_nano_runs(
            ax,
            nano_runs,
            f"{dataset}/{metric}",
            baselines,
            baselines_std,
            show_legend=False,
            show_xlabel=False,
            show_ylabel=(j == 0),
        )
        ax.set_title(dataset)
    fig.supxlabel("Training Time (seconds)")

    for ax in axs.flatten():
        font_size = fig.texts[-1].get_fontsize()
        ax.xaxis.label.set_size(font_size)
        ax.yaxis.label.set_size(font_size)

    legend_handles_labels = [list(zip(*ax.get_legend_handles_labels())) for ax in axs.flatten()]
    legend_handles_labels = functools.reduce(lambda a, b: a + b, legend_handles_labels)
    unique = {label: handle for (handle, label) in legend_handles_labels}
    fig.legend(unique.values(), unique.keys(), loc="outside upper center", ncol=3)
    return fig, axs
