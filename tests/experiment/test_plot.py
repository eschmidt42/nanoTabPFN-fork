import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from nanotabpfn.experiment.plot import plot_nano_runs, plot_run_grid


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


def _make_polars_baselines() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "baseline": ["Baseline"],
            "iris/ROC AUC": [0.75],
            "wine/ROC AUC": [0.65],
        }
    )


def test_plot_nano_runs_interpolates_runs_and_plots_baseline_uncertainty() -> None:
    fig, ax = plt.subplots()

    plot_nano_runs(
        ax,
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

    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        np.asarray(ax.lines[0].get_ydata(), dtype=np.float64),
        [0.5, 0.6, 0.9],
    )
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), [0.0, 2.0])
    np.testing.assert_allclose(
        np.asarray(ax.lines[1].get_ydata(), dtype=np.float64),
        [0.75, 0.75],
    )
    assert len(ax.collections) == 2
    assert ax.get_xlabel() == "Training time (seconds)"
    assert ax.get_ylabel() == "ROC AUC"
    assert ax.get_xlim() == (0.0, 2.0)
    assert ax.get_ylim()[1] == 1.0
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [
        "nanoTabPFN",
        "Baseline",
    ]

    plt.close(fig)


def test_plot_nano_runs_honors_hidden_label_legend_and_tick_options() -> None:
    fig, ax = plt.subplots()

    plot_nano_runs(
        ax,
        _make_polars_runs(),
        "iris/ROC AUC",
        show_legend=False,
        show_xlabel=False,
        show_ylabel=False,
        show_xtics=False,
    )

    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == ""
    assert ax.get_legend() is None
    assert all(tick.tick1line.get_markersize() == 0 for tick in ax.xaxis.get_major_ticks())

    plt.close(fig)


def test_plot_run_grid_creates_dataset_panels_and_combined_legend() -> None:
    fig, axs = plot_run_grid(_make_polars_runs(), _make_polars_baselines())

    assert len(axs) == 2
    assert {ax.get_title() for ax in axs} == {"iris", "wine"}
    assert axs[0].get_ylabel() == "ROC AUC"
    assert axs[1].get_ylabel() == ""
    assert fig.supxlabel("Training Time (seconds)") is not None
    assert len(fig.legends) == 1
    assert {text.get_text() for text in fig.legends[0].get_texts()} == {
        "nanoTabPFN",
        "Baseline",
    }

    plt.close(fig)
