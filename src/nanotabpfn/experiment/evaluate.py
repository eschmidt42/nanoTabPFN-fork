from collections.abc import Mapping
from typing import Protocol, Self

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold


class ProbabilisticClassifier(Protocol):
    def fit(self, X: NDArray, y: NDArray) -> Self: ...

    def predict_proba(self, X: NDArray) -> NDArray: ...


SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


type EvalResults = dict[str, float]


def evaluate_model(
    model: ProbabilisticClassifier,
    datasets: Mapping[str, tuple[NDArray, NDArray]],
    cv: BaseCrossValidator = SKF,
) -> EvalResults:
    """Evaluates a model on multiple datasets and returns metrics"""
    metrics = {}
    for dataset_name, (X, y) in datasets.items():
        targets = []
        probabilities = []

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            targets.append(y_test)
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:  # binary classification with neural network
                y_proba = y_proba[:, 1]
            probabilities.append(y_proba)

        targets = np.concatenate(targets, axis=0)
        probabilities = np.concatenate(probabilities, axis=0)

        metrics[f"{dataset_name}/ROC AUC"] = roc_auc_score(
            targets, probabilities, multi_class="ovr"
        )

    metric_names = list({key.split("/")[-1] for key in metrics})
    for metric_name in metric_names:
        avg_metric = np.mean([metrics[key] for key in metrics if key.endswith(metric_name)])
        metrics[f"{metric_name}"] = float(avg_metric)

    return metrics
