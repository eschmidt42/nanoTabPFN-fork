from typing import TypedDict

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from nanotabpfn import NanoTabPFNModel, trainer


class TrainingBatch(TypedDict):
    train_test_split_index: int
    x: torch.Tensor
    y: torch.Tensor


class TrainingBatchDataset(Dataset[TrainingBatch]):
    def __init__(self, batches: list[TrainingBatch]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index: int) -> TrainingBatch:
        return self.batches[index]


class TinyTrainModel(NanoTabPFNModel):
    def __init__(self) -> None:
        super().__init__(
            embedding_size=2,
            num_attention_heads=1,
            mlp_hidden_size=2,
            num_layers=0,
            num_outputs=2,
        )
        self.class_logits = nn.Parameter(torch.zeros(2))
        self.inputs_seen: list[tuple[torch.Tensor, torch.Tensor, int]] = []

    def forward(
        self,
        features_and_targets: tuple[torch.Tensor, torch.Tensor],
        train_test_split_index: int,
    ) -> torch.Tensor:
        features, targets = features_and_targets
        self.inputs_seen.append(
            (features.detach().clone(), targets.detach().clone(), train_test_split_index)
        )
        test_features = features[:, train_test_split_index:, :1]
        return test_features * self.class_logits.view(1, 1, -1)


def _make_batch() -> TrainingBatch:
    return {
        "train_test_split_index": 2,
        "x": torch.tensor(
            [
                [[0.1], [0.2], [2.0], [-1.0]],
                [[-0.1], [0.3], [1.0], [-2.0]],
            ]
        ),
        "y": torch.tensor([[0, 1, 1, 0], [1, 0, 1, 0]]),
    }


def test_train_updates_model_with_training_rows_and_test_targets(monkeypatch: pytest.MonkeyPatch):
    batch = _make_batch()
    model = TinyTrainModel()
    original_logits = model.class_logits.detach().clone()
    original_cross_entropy: type[nn.CrossEntropyLoss] = nn.CrossEntropyLoss

    class RecordingCrossEntropyLoss(original_cross_entropy):
        def __init__(self) -> None:
            super().__init__()
            self.inputs_seen: torch.Tensor | None = None
            self.targets_seen: torch.Tensor | None = None
            recorded_losses.append(self)

        def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            self.inputs_seen = input.detach().clone()
            self.targets_seen = target.detach().clone()
            return super().forward(input, target)

    recorded_losses: list[RecordingCrossEntropyLoss] = []

    monkeypatch.setattr(trainer.nn, "CrossEntropyLoss", RecordingCrossEntropyLoss)

    trained_model, history = trainer.train(
        model,
        DataLoader(TrainingBatchDataset([batch]), batch_size=None),
        device=torch.device("cpu"),
    )

    assert trained_model is model
    assert history == []
    torch.testing.assert_close(model.inputs_seen[0][0], batch["x"])
    torch.testing.assert_close(model.inputs_seen[0][1], batch["y"][:, :2])
    assert model.inputs_seen[0][2] == 2
    assert recorded_losses[0].targets_seen is not None
    assert recorded_losses[0].inputs_seen is not None
    assert recorded_losses[0].targets_seen.tolist() == [1, 0, 1, 0]
    assert recorded_losses[0].inputs_seen.shape == (4, 2)
    assert not torch.equal(model.class_logits.detach(), original_logits)


def test_train_evaluates_at_configured_interval_and_restores_training_mode(
    capsys: pytest.CaptureFixture[str],
):
    model = TinyTrainModel()
    classifiers: list[trainer.NanoTabPFNClassifier] = []
    scores = {"accuracy": 0.75}

    def eval_func(classifier: trainer.NanoTabPFNClassifier) -> dict[str, float]:
        classifiers.append(classifier)
        return scores

    trained_model, history = trainer.train(
        model,
        DataLoader(TrainingBatchDataset([_make_batch() for _ in range(3)]), batch_size=None),
        device=torch.device("cpu"),
        steps_per_eval=2,
        eval_func=eval_func,
    )

    assert trained_model is model
    assert len(classifiers) == 1
    assert classifiers[0].model is model
    assert classifiers[0].device == torch.device("cpu")
    assert len(history) == 1
    elapsed_time, evaluated_scores = history[0]
    assert elapsed_time >= 0
    assert evaluated_scores == scores
    assert model.training

    output = capsys.readouterr().out
    assert "accuracy  0.7500" in output
    assert output.count("loss") == 1


def test_train_without_eval_func_prints_loss_only_at_configured_interval(
    capsys: pytest.CaptureFixture[str],
):
    model = TinyTrainModel()

    trained_model, history = trainer.train(
        model,
        DataLoader(TrainingBatchDataset([_make_batch()]), batch_size=None),
        device=torch.device("cpu"),
        steps_per_eval=1,
    )

    assert trained_model is model
    assert history == []
    output = capsys.readouterr().out
    assert "loss" in output
    assert "accuracy" not in output


@pytest.mark.parametrize(
    ("requested_device", "expected_device", "uses_default_device"),
    [
        (torch.device("cpu"), torch.device("cpu"), False),
        (None, torch.device("cpu"), True),
    ],
)
def test_train_uses_requested_or_default_device(
    requested_device: torch.device | None,
    expected_device: torch.device,
    uses_default_device: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    default_device_calls: list[bool] = []

    def get_default_device() -> torch.device:
        default_device_calls.append(True)
        return expected_device

    monkeypatch.setattr(trainer, "get_default_device", get_default_device)
    model = TinyTrainModel()
    to_calls: list[torch.device] = []
    original_to = model.to

    def recording_to(device: torch.device) -> TinyTrainModel:
        to_calls.append(device)
        return original_to(device)

    monkeypatch.setattr(model, "to", recording_to)

    trained_model, history = trainer.train(
        model, DataLoader(TrainingBatchDataset([]), batch_size=None), device=requested_device
    )

    assert trained_model is model
    assert history == []
    assert to_calls == [expected_device]
    assert bool(default_device_calls) is uses_default_device
