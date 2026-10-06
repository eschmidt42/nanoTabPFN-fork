import numpy as np
import pytest
import torch
from torch import nn

from nanotabpfn.model import (
    Decoder,
    FeatureEncoder,
    NanoTabPFNClassifier,
    NanoTabPFNModel,
    TargetEncoder,
    TransformerEncoderLayer,
)


def test_feature_encoder_normalizes_using_training_rows_and_clips():
    encoder = FeatureEncoder(embedding_size=1)
    with torch.no_grad():
        encoder.linear_layer.weight.fill_(1)
        encoder.linear_layer.bias.zero_()

    features = torch.tensor([[[0.0], [2.0], [1000.0]]])

    encoded = encoder(features, train_test_split_index=2)

    expected = torch.tensor([[[[-(2**-0.5)]], [[2**-0.5]], [[100.0]]]])
    assert encoded.shape == (1, 3, 1, 1)
    torch.testing.assert_close(encoded, expected)


def test_target_encoder_pads_each_batch_with_its_training_target_mean():
    encoder = TargetEncoder(embedding_size=1)
    with torch.no_grad():
        encoder.linear_layer.weight.fill_(1)
        encoder.linear_layer.bias.zero_()

    targets = torch.tensor([[[1.0], [3.0]], [[2.0], [6.0]]])

    encoded = encoder(targets, num_rows=4)

    expected = torch.tensor(
        [
            [[[1.0]], [[3.0]], [[2.0]], [[2.0]]],
            [[[2.0]], [[6.0]], [[4.0]], [[4.0]]],
        ]
    )
    assert encoded.shape == (2, 4, 1, 1)
    torch.testing.assert_close(encoded, expected)


def test_transformer_encoder_layer_does_not_let_training_rows_attend_to_test_rows():
    layer = TransformerEncoderLayer(embedding_size=4, nhead=2, mlp_hidden_size=8)
    layer.eval()
    inputs = torch.randn(2, 5, 3, 4)
    changed_test_inputs = inputs.clone()
    changed_test_inputs[:, 3:] += 100

    original_output = layer(inputs, train_test_split_index=3)
    changed_output = layer(changed_test_inputs, train_test_split_index=3)

    assert original_output.shape == inputs.shape
    assert torch.isfinite(original_output).all()
    torch.testing.assert_close(original_output[:, :3], changed_output[:, :3])


def test_decoder_returns_requested_number_of_outputs():
    decoder = Decoder(embedding_size=4, mlp_hidden_size=8, num_outputs=3)

    output = decoder(torch.randn(2, 5, 4))

    assert output.shape == (2, 5, 3)


@pytest.mark.parametrize("target_has_trailing_dimension", [False, True])
def test_model_forward_returns_logits_for_test_rows(target_has_trailing_dimension: bool):
    model = NanoTabPFNModel(
        embedding_size=4,
        num_attention_heads=2,
        mlp_hidden_size=8,
        num_layers=1,
        num_outputs=3,
    )
    features = torch.randn(2, 5, 2)
    targets = torch.randn(2, 3)
    if target_has_trailing_dimension:
        targets = targets.unsqueeze(-1)

    output = model((features, targets), train_test_split_index=3)

    assert output.shape == (2, 2, 3)
    assert torch.isfinite(output).all()


class FixedLogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.tensor([[0.0, 2.0, 20.0], [4.0, 1.0, -20.0]])

    def forward(self, features_and_targets, train_test_split_index):
        num_test_rows = features_and_targets[0].shape[1] - train_test_split_index
        return self.logits[:num_test_rows].unsqueeze(0)


def test_classifier_predict_proba_trims_missing_classes_and_normalizes():
    classifier = NanoTabPFNClassifier(FixedLogitModel(), device=torch.device("cpu"))  # ty: ignore[invalid-argument-type]
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0, 1, 0])
    X_test = np.array([[3.0], [4.0]])
    classifier.fit(X_train, y_train)

    probabilities = classifier.predict_proba(X_test)

    expected = torch.softmax(torch.tensor([[0.0, 2.0], [4.0, 1.0]]), dim=1).numpy()
    assert probabilities.shape == (2, 2)
    np.testing.assert_allclose(probabilities, expected)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))
    np.testing.assert_array_equal(classifier.predict(X_test), np.array([1, 0]))
