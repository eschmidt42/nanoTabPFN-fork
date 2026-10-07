import numpy as np
import polars as pl

from nanotabpfn.experiment.data import (
    _get_column_masks,
    _get_feature_preprocessor,
    _to_polars_frame,
)


def test_to_polars_frame_infers_each_column_and_preserves_order() -> None:
    X = np.array([["1", 2.5, "red"], ["2", 4.5, "blue"]], dtype=object)

    frame = _to_polars_frame(X)

    assert frame.columns == ["0", "1", "2"]
    assert frame.dtypes == [pl.String, pl.Float64, pl.String]
    assert frame["0"].to_list() == ["1", "2"]
    assert frame["1"].to_list() == [2.5, 4.5]
    assert frame["2"].to_list() == ["red", "blue"]


def test_to_polars_frame_normalizes_none_and_nan_to_null() -> None:
    X = np.array(
        [["1", "red", None], [np.nan, None, np.nan], ["3", "blue", None]],
        dtype=object,
    )

    frame = _to_polars_frame(X)

    assert frame.dtypes == [pl.String, pl.String, pl.Null]
    assert frame["0"].to_list() == ["1", None, "3"]
    assert frame["1"].to_list() == ["red", None, "blue"]
    assert frame["2"].null_count() == 3


def test_get_column_masks_classifies_numeric_and_categorical_columns() -> None:
    frame = pl.DataFrame(
        {
            "numeric": [1.0, None, 3.0],
            "numeric_strings": ["1", None, "3"],
            "categorical": ["red", None, "blue"],
            "partly_numeric": ["1", "invalid", None],
        }
    )

    num_mask, cat_mask = _get_column_masks(frame)

    np.testing.assert_array_equal(num_mask, [True, True, False, False])
    np.testing.assert_array_equal(cat_mask, [False, False, True, True])
    assert num_mask.dtype == np.bool_
    assert cat_mask.dtype == np.bool_


def test_get_column_masks_excludes_constant_and_all_null_columns() -> None:
    frame = pl.DataFrame(
        {
            "varying_numeric": [1, 2, 3],
            "constant": [7, 7, None],
            "all_null": [None, None, None],
        }
    )

    num_mask, cat_mask = _get_column_masks(frame)

    np.testing.assert_array_equal(num_mask, [True, False, False])
    np.testing.assert_array_equal(cat_mask, [False, False, False])


def test_get_feature_preprocessor_converts_numeric_strings_and_preserves_nans() -> None:
    X = np.array([["1", "2.5"], ["2", np.nan], [np.nan, "4.5"]], dtype=object)

    transformed = _get_feature_preprocessor(X).fit_transform(X)

    np.testing.assert_allclose(
        transformed,
        np.array([[1.0, 2.5], [2.0, np.nan], [np.nan, 4.5]]),
        equal_nan=True,
    )


def test_get_feature_preprocessor_encodes_categories_and_unknown_values_as_nan() -> None:
    X = np.array([["red"], ["blue"], ["red"], [np.nan]], dtype=object)
    preprocessor = _get_feature_preprocessor(X)

    transformed = preprocessor.fit_transform(X)
    unknown = preprocessor.transform(np.array([["green"]], dtype=object))

    np.testing.assert_array_equal(np.unique(transformed[:3]), np.array([0.0, 1.0]))
    assert np.isnan(transformed[3, 0])
    assert np.isnan(unknown[0, 0])


def test_get_feature_preprocessor_drops_constant_and_all_missing_columns() -> None:
    X = np.array(
        [
            [1, 7, np.nan, "red"],
            [2, 7, np.nan, "blue"],
            [3, 7, np.nan, "red"],
        ],
        dtype=object,
    )

    transformed = _get_feature_preprocessor(X).fit_transform(X)

    assert transformed.shape == (3, 2)
    np.testing.assert_array_equal(transformed[:, 0], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(np.unique(transformed[:, 1]), np.array([0.0, 1.0]))
