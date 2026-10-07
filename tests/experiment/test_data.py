import numpy as np

from nanotabpfn.experiment.data import _get_feature_preprocessor


def test_get_feature_preprocessor_converts_numeric_strings_and_preserves_nans() -> None:
    X = np.array([["1", "2.5"], ["2", np.nan], ["3", "4.5"]], dtype=object)

    transformed = _get_feature_preprocessor(X).fit_transform(X)

    np.testing.assert_allclose(
        transformed,
        np.array([[1.0, 2.5], [2.0, np.nan], [3.0, 4.5]]),
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
