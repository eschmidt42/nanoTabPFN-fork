import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from plotnine import ggplot

from nanotabpfn.experiment import plot_multiple_summary_metrics as exported_plot_run_grid
from nanotabpfn.experiment import plot_single_summary_metric as exported_plot_nano_runs
from nanotabpfn.experiment.plot import (
    XTypesEnum,
    _align_nano_runs_by_iteration,
    _align_nano_runs_by_training_time,
    _get_data_to_plot_nano_runs,
    _get_nano_runs_iteration_summary,
    _get_nano_runs_time_summary,
    plot_multiple_summary_metrics,
    plot_single_summary_metric,
)


def _make_polars_runs() -> list[pl.DataFrame]:
    return [
        pl.DataFrame(
            {
                "training_time": [0.0, 2.0],
                "iris/ROC AUC": [0.4, 0.8],
                "wine/ROC AUC": [0.3, 0.7],
            }
        ),
        pl.DataFrame(
            {
                "training_time": [0.0, 1.0, 2.0],
                "iris/ROC AUC": [0.6, 0.6, 1.0],
                "wine/ROC AUC": [0.5, 0.6, 0.9],
            }
        ),
    ]


def _make_polars_runs_with_iterations() -> list[pl.DataFrame]:
    return [
        pl.DataFrame(
            {
                "training_time": [0.0, 2.0],
                "iteration": [0, 2],
                "iris/ROC AUC": [0.4, 0.8],
                "wine/ROC AUC": [0.3, 0.7],
            }
        ),
        pl.DataFrame(
            {
                "training_time": [0.0, 1.0, 2.0],
                "iteration": [0, 1, 2],
                "iris/ROC AUC": [0.6, 0.6, 1.0],
                "wine/ROC AUC": [0.5, 0.6, 0.9],
            }
        ),
    ]


def _make_polars_baselines() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "baseline": ["Baseline"],
            "iris/ROC AUC": [0.75],
            "wine/ROC AUC": [0.65],
        }
    )


def _make_interpolation_runs() -> list[pl.DataFrame]:
    return [
        pl.DataFrame(
            {
                "training_time": [2.0, 0.0],
                "ROC AUC": [0.8, 0.2],
            }
        ),
        pl.DataFrame(
            {
                "training_time": [2.0, 1.0, 0.0],
                "ROC AUC": [0.6, float("nan"), 0.4],
            }
        ),
    ]


def test_align_nano_runs_by_training_time_interpolates_and_sorts_runs() -> None:
    aligned, run_columns = _align_nano_runs_by_training_time(_make_interpolation_runs(), "ROC AUC")

    assert run_columns == ["run_0", "run_1"]
    np.testing.assert_array_equal(aligned["training_time"].to_numpy(), [0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        aligned.select(run_columns).to_numpy(),
        [[0.2, 0.4], [0.5, 0.5], [0.8, 0.6]],
    )


def test_align_nano_runs_by_iteration_interpolates_and_sorts_runs() -> None:
    aligned, run_columns = _align_nano_runs_by_iteration(
        _make_polars_runs_with_iterations(),
        "iris/ROC AUC",
    )

    assert run_columns == ["run_0", "run_1"]
    np.testing.assert_array_equal(aligned["iteration"].to_numpy(), [0, 1, 2])
    np.testing.assert_allclose(
        aligned.select(run_columns).to_numpy(),
        [[0.4, 0.6], [0.6, 0.6], [0.8, 1.0]],
    )


def test_get_nano_runs_time_summary_returns_mean_and_sample_std() -> None:
    training_times, mean, std = _get_nano_runs_time_summary(
        _make_interpolation_runs(),
        "ROC AUC",
    )

    np.testing.assert_array_equal(training_times, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(mean, [0.3, 0.5, 0.7])
    assert std is not None
    np.testing.assert_allclose(std, [np.sqrt(0.02), 0.0, np.sqrt(0.02)])


def test_get_nano_runs_iteration_summary_returns_mean_and_sample_std() -> None:
    iterations, mean, std = _get_nano_runs_iteration_summary(
        _make_polars_runs_with_iterations(),
        "iris/ROC AUC",
    )

    np.testing.assert_array_equal(iterations, [0, 1, 2])
    np.testing.assert_allclose(mean, [0.5, 0.6, 0.9])
    assert std is not None
    np.testing.assert_allclose(std, [np.sqrt(0.02), 0.0, np.sqrt(0.02)], atol=1e-15)


def test_get_nano_runs_iteration_summary_returns_no_std_for_one_run() -> None:
    iterations, mean, std = _get_nano_runs_iteration_summary(
        [_make_polars_runs_with_iterations()[0]],
        "iris/ROC AUC",
    )

    np.testing.assert_array_equal(iterations, [0, 2])
    np.testing.assert_allclose(mean, [0.4, 0.8])
    assert std is None


def test_get_nano_runs_time_summary_returns_no_std_for_one_run() -> None:
    training_times, mean, std = _get_nano_runs_time_summary(
        [_make_interpolation_runs()[0]],
        "ROC AUC",
    )

    np.testing.assert_array_equal(training_times, [0.0, 2.0])
    np.testing.assert_allclose(mean, [0.2, 0.8])
    assert std is None


def test_plot_single_summary_metric_returns_plotnine_plot_with_summary_and_baselines() -> None:
    line_data, ribbon_data = _get_data_to_plot_nano_runs(
        _make_polars_runs(),
        "iris/ROC AUC",
        _make_polars_baselines(),
        pl.DataFrame({"baseline": ["Baseline"], "iris/ROC AUC": [0.05]}),
        XTypesEnum.training_time,
    )
    plot = plot_single_summary_metric(
        _make_polars_runs(),
        "iris/ROC AUC",
        baselines=_make_polars_baselines(),
        baselines_std=pl.DataFrame(
            {
                "baseline": ["Baseline"],
                "iris/ROC AUC": [0.05],
            }
        ),
    )

    assert isinstance(plot, ggplot)
    nano_data = line_data[line_data["series"] == "nanoTabPFN"]
    np.testing.assert_array_equal(nano_data["training_time"], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(nano_data["value"], [0.5, 0.6, 0.9])

    baseline_ribbon = ribbon_data[ribbon_data["series"] == "Baseline"]
    np.testing.assert_allclose(baseline_ribbon["lower"], [0.7, 0.7])
    np.testing.assert_allclose(baseline_ribbon["upper"], [0.8, 0.8])
    nano_ribbon = ribbon_data[ribbon_data["series"] == "nanoTabPFN"]
    np.testing.assert_allclose(
        nano_ribbon["lower"],
        [0.5 - np.sqrt(0.02), 0.6, 0.9 - np.sqrt(0.02)],
    )
    np.testing.assert_allclose(
        nano_ribbon["upper"],
        [0.5 + np.sqrt(0.02), 0.6, 0.9 + np.sqrt(0.02)],
    )
    assert plot.labels.x == "Training time (seconds)"
    assert plot.labels.y == "ROC AUC"
    plt.close(plot.draw())


def test_plot_single_summary_metric_uses_training_iteration_axis() -> None:
    line_data, ribbon_data = _get_data_to_plot_nano_runs(
        _make_polars_runs_with_iterations(),
        "iris/ROC AUC",
        _make_polars_baselines(),
        pl.DataFrame({"baseline": ["Baseline"], "iris/ROC AUC": [0.05]}),
        XTypesEnum.iteration,
    )
    plot = plot_single_summary_metric(
        _make_polars_runs_with_iterations(),
        "iris/ROC AUC",
        baselines=_make_polars_baselines(),
        baselines_std=pl.DataFrame({"baseline": ["Baseline"], "iris/ROC AUC": [0.05]}),
        x_type=XTypesEnum.iteration,
    )

    assert "training_time" not in line_data.columns
    nano_data = line_data[line_data["series"] == "nanoTabPFN"]
    np.testing.assert_array_equal(nano_data["iteration"], [0, 1, 2])
    np.testing.assert_allclose(nano_data["value"], [0.5, 0.6, 0.9])
    baseline_ribbon = ribbon_data[ribbon_data["series"] == "Baseline"]
    np.testing.assert_array_equal(baseline_ribbon["iteration"], [0.0, 2.0])
    assert plot.labels.x == "Training iteration"
    plt.close(plot.draw())


def test_plot_single_summary_metric_honors_label_and_tick_options() -> None:
    plot = plot_single_summary_metric(
        _make_polars_runs(),
        "iris/ROC AUC",
        show_legend=False,
        show_xlabel=False,
        show_ylabel=False,
        show_xtics=False,
    )

    assert plot.labels.x is None
    assert plot.labels.y is None
    fig = plot.draw()
    assert fig.axes[0].get_xlabel() == ""
    assert fig.axes[0].get_ylabel() == ""
    assert fig.axes[0].get_legend() is None
    assert all(not label.get_visible() for label in fig.axes[0].get_xticklabels())
    plt.close(fig)


def test_plot_multiple_summary_metrics_uses_training_iteration_axis() -> None:
    plot = plot_multiple_summary_metrics(
        _make_polars_runs_with_iterations(),
        x_type=XTypesEnum.iteration,
    )

    assert isinstance(plot, ggplot)
    assert plot.labels.x == "Training iteration"
    fig = plot.draw()
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_multiple_summary_metrics_facets_datasets_and_is_exported() -> None:
    plot = plot_multiple_summary_metrics(_make_polars_runs(), _make_polars_baselines())

    assert isinstance(plot, ggplot)
    assert isinstance(exported_plot_nano_runs(_make_polars_runs(), "iris/ROC AUC"), ggplot)
    assert isinstance(exported_plot_run_grid(_make_polars_runs()), ggplot)
    fig = plot.draw()
    assert len(fig.axes) == 2
    plt.close(fig)
