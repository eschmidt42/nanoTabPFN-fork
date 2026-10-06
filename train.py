from collections.abc import Sequence
from typing import TypedDict

from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from nanotabpfn import (
    NanoTabPFNClassifier,
    NanoTabPFNModel,
    PriorDumpDataLoader,
    get_default_device,
    set_randomness_seed,
    train,
)


class Scores(TypedDict):
    roc_auc: float
    acc: float
    balanced_acc: float


def eval(
    classifier: NanoTabPFNClassifier, datasets: Sequence[tuple[NDArray, NDArray, NDArray, NDArray]]
) -> Scores:

    accumulating_scores: Scores = {"roc_auc": 0.0, "acc": 0.0, "balanced_acc": 0.0}

    for X_train, X_test, y_train, y_test in datasets:
        classifier.fit(X_train, y_train)

        prob = classifier.predict_proba(X_test)
        pred = prob.argmax(axis=1)  # avoid a second forward pass by not calling predict

        if prob.shape[1] == 2:
            prob = prob[:, 1]

        accumulating_scores["roc_auc"] += float(roc_auc_score(y_test, prob, multi_class="ovr"))
        accumulating_scores["acc"] += float(accuracy_score(y_test, pred))
        accumulating_scores["balanced_acc"] += float(balanced_accuracy_score(y_test, pred))

    n = len(datasets)

    average_scores: Scores = {
        "roc_auc": accumulating_scores["roc_auc"] / n,
        "acc": accumulating_scores["acc"] / n,
        "balanced_acc": accumulating_scores["balanced_acc"] / n,
    }
    return average_scores


if __name__ == "__main__":
    set_randomness_seed(0)

    datasets = []
    datasets.append(
        train_test_split(*load_breast_cancer(return_X_y=True), test_size=0.5, random_state=0)
    )

    device = get_default_device()
    model = NanoTabPFNModel(
        embedding_size=96, num_attention_heads=4, mlp_hidden_size=192, num_layers=3, num_outputs=2
    )
    prior = PriorDumpDataLoader("300k_150x5_2.h5", num_steps=2500, batch_size=32, device=device)
    model, history = train(model, prior, lr=4e-3, steps_per_eval=25)
    print("Final evaluation:")
    print(eval(NanoTabPFNClassifier(model, device), datasets))
