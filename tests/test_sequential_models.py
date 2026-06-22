"""Tests for row-as-timestep sequential image models."""

import numpy as np
import tensorflow as tf

from models.sequential import build_bilstm, build_lstm, build_rnn


def test_bilstm_binary_shape_and_probabilities():
    model = build_bilstm(num_classes=1, hidden_units=8)
    assert model.name == "bilstm_sequence"
    assert model.input_shape == (None, 32, 96)
    assert model.output_shape == (None, 1)

    y = model(np.zeros((2, 32, 96), dtype=np.float32), training=False).numpy()
    assert y.shape == (2, 1)
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 1.0


def test_bilstm_multiclass_shape_and_softmax():
    model = build_bilstm(num_classes=20, hidden_units=8)
    assert model.output_shape == (None, 20)

    y = model(np.zeros((2, 32, 96), dtype=np.float32), training=False).numpy()
    np.testing.assert_allclose(y.sum(axis=1), np.ones(2), atol=1e-6)


def test_bilstm_can_include_masking_layer():
    model = build_bilstm(num_classes=20, hidden_units=8, use_masking=True)
    assert any(isinstance(layer, tf.keras.layers.Masking) for layer in model.layers)


def test_rnn_and_lstm_build_multiclass_models():
    rnn = build_rnn(num_classes=20, hidden_units=8)
    lstm = build_lstm(num_classes=20, hidden_units=8)

    assert rnn.input_shape == (None, 32, 96)
    assert lstm.input_shape == (None, 32, 96)
    assert rnn.output_shape == (None, 20)
    assert lstm.output_shape == (None, 20)
