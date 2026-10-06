import functools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_runs(
    ax: plt.Axes,
    runs: list[pd.DataFrame],
    metric: str,
    baselines: pd.DataFrame | None = None,
    baselines_std: pd.DataFrame | None = None,
    show_legend: bool = True,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_xtics: bool = True,
):
    """
    Plots the run for a given metric and adds baselines

    `runs` is a list of dataframes where each dataframe corresponds
    to a run from a model with the same config but a different seed.
    Each dataframe needs to have a `"training_time"` column and a `metric`column.

    `baselines` is a DataFrame with metric columns and rows whose index correspond to
    a ML algorithm.
    """
    colors = sns.color_palette("tab10")[1:]
    linestyles = [
        "--",  # dashed
        "-.",  # dash-dot
        ":",  # dotted
        (0, (3, 1, 1, 1)),  # dash-dot-dot
        (0, (5, 5)),  # spaced dash
    ]

    training_times = [run["training_time"].tolist() for run in runs]
    training_times = sorted({item for sublist in training_times for item in sublist})
    shared_time_runs = []
    for run in runs:
        run = run.copy()
        run = run[[metric, "training_time"]].set_index("training_time").reindex(training_times)
        run = run.interpolate()
        shared_time_runs.append(run)
    all_runs = pd.concat(shared_time_runs, axis=1).dropna()

    # plot mean and std of all runs or single run if only one run
    mean = all_runs.mean(axis=1)
    ax.plot(mean.index, mean, label="nanoTabPFN", zorder=2, color="blue")
    if all_runs.shape[1] > 1:  # more than one run
        std = all_runs.std(axis=1)
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.2, zorder=2)

    # plot horizontal lines for baselines
    if baselines is not None:
        for i, (baseline_name, baseline_value) in enumerate(baselines[metric].items()):
            # draw a horizontal line that ends at the same x as the runs
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
            if baselines_std is not None and baseline_name in baselines_std.index:
                std = baselines_std.loc[baseline_name, metric]
                ax.fill_between(
                    [0, max(training_times)],
                    [baseline_value - std, baseline_value - std],
                    [baseline_value + std, baseline_value + std],
                    alpha=0.2,
                    zorder=1,
                )

    # Plot Style
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

    # order legend entries by their y-value at the end of the plot
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

    # remove border
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_run_grid(
    runs: list[pd.DataFrame],
    baselines: pd.DataFrame | None = None,
    baselines_std: pd.DataFrame | None = None,
):
    """Plots the runs in a grid of metrics x datasets"""
    # drop all columns without "/" for dataset/metric format
    datasets = list({col.split("/")[0] for col in runs[0].columns if "/" in col})
    metric = "ROC AUC"
    figsize = (len(datasets) * 4, 4.6)
    fig, axs = plt.subplots(
        1, len(datasets), figsize=figsize, sharex=True, sharey=True, layout="constrained"
    )
    fig.set_layout_engine("constrained", w_pad=0.0, h_pad=0.1)
    # Plot each metric and dataset
    for j, dataset in enumerate(datasets):
        ax = axs[j]
        plot_runs(
            ax,
            runs,
            f"{dataset}/{metric}",
            baselines,
            baselines_std,
            show_legend=False,
            show_xlabel=False,
            show_ylabel=(j == 0),
        )
        ax.set_title(dataset)
    fig.supxlabel("Training Time (seconds)")

    # y-axis and x-axis labels should have the same size as supxlabel
    for ax in axs.flatten():
        font_size = fig.texts[-1].get_fontsize()
        ax.xaxis.label.set_size(font_size)
        ax.yaxis.label.set_size(font_size)

    # Create a single legend for the entire figure
    legend_handels_labels = [list(zip(*ax.get_legend_handles_labels())) for ax in axs.flatten()]
    legend_handels_labels = functools.reduce(lambda a, b: a + b, legend_handels_labels)
    unique = {label: handle for (handle, label) in legend_handels_labels}
    fig.legend(unique.values(), unique.keys(), loc="outside upper center", ncol=3)
    return fig, axs
