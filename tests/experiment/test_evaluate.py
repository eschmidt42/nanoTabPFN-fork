from typing import Self

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import KFold

from nanotabpfn.experiment.evaluate import evaluate_model


class FeatureProbabilityClassifier:
    def __init__(self) -> None:
        self.fit_calls: list[tuple[NDArray, NDArray]] = []
        self.predicted_inputs: list[NDArray] = []

    def fit(self, X: NDArray, y: NDArray) -> Self:
        self.fit_calls.append((X.copy(), y.copy()))
        return self

    def predict_proba(self, X: NDArray) -> NDArray:
        self.predicted_inputs.append(X.copy())
        return X.copy()


def test_evaluate_model_scores_multiple_binary_datasets_and_averages_scores():
    targets = np.array([0, 1, 0, 1])
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8],
        ]
    )
    datasets = {
        "perfect": (probabilities, targets),
        "inverse": (probabilities[:, ::-1].copy(), targets),
    }
    classifier = FeatureProbabilityClassifier()

    results = evaluate_model(classifier, datasets, cv=KFold(n_splits=2))

    assert results == {
        "perfect/ROC AUC": 1.0,
        "inverse/ROC AUC": 0.0,
        "ROC AUC": 0.5,
    }
    assert len(classifier.fit_calls) == 4
    assert len(classifier.predicted_inputs) == 4
    np.testing.assert_array_equal(classifier.fit_calls[0][0], probabilities[2:])
    np.testing.assert_array_equal(classifier.fit_calls[0][1], targets[2:])
    np.testing.assert_array_equal(classifier.predicted_inputs[0], probabilities[:2])
    np.testing.assert_array_equal(classifier.fit_calls[1][0], probabilities[:2])
    np.testing.assert_array_equal(classifier.fit_calls[1][1], targets[:2])
    np.testing.assert_array_equal(classifier.predicted_inputs[1], probabilities[2:])


def test_evaluate_model_scores_multiclass_probabilities():
    targets = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.eye(3)[targets]
    classifier = FeatureProbabilityClassifier()

    results = evaluate_model(
        classifier,
        {"multiclass": (probabilities, targets)},
        cv=KFold(n_splits=2),
    )

    assert results == {"multiclass/ROC AUC": 1.0, "ROC AUC": 1.0}
    assert len(classifier.fit_calls) == 2
    assert len(classifier.predicted_inputs) == 2
    assert all(predictions.shape[1] == 3 for predictions in classifier.predicted_inputs)
