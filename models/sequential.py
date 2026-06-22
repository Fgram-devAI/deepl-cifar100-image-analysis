"""Sequential models: vanilla RNN, LSTM, and Bi-LSTM with optional masking."""

from typing import Literal
import tensorflow as tf

keras = tf.keras
layers = tf.keras.layers


def _head_units_and_activation(
    *, head: str, num_classes: int | None
) -> tuple[int, str]:
    """Resolve legacy head names and newer num_classes routing."""
    if num_classes is not None:
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        return (1, "sigmoid") if num_classes == 1 else (num_classes, "softmax")
    if head == "binary":
        return 1, "sigmoid"
    if head == "multiclass":
        return 10, "softmax"
    raise ValueError("head must be 'binary' or 'multiclass'")


def _input_layers(use_masking: bool) -> tuple[keras.Input, tf.Tensor]:
    inputs = keras.Input(shape=(32, 96), name="sequence_input")
    x: tf.Tensor = inputs
    if use_masking:
        x = layers.Masking(mask_value=0.0, name="row_masking")(x)
    return inputs, x


def build_rnn(
    activation: str = "tanh",
    use_masking: bool = False,
    head: str = "binary",
    hidden_units: int = 64,
    dropout: float = 0.2,
    num_classes: int | None = None,
) -> keras.Model:
    """
    Build a vanilla RNN model for sequence classification.

    Args:
        activation: Recurrent activation ("tanh" or "relu").
        use_masking: If True, add a Masking layer before the RNN.
        head: Legacy output head ("binary" or "multiclass").
        hidden_units: Number of hidden units in the RNN.
        dropout: Dropout rate.
        num_classes: Preferred output size. ``1`` gives a sigmoid binary head;
            values above one give a softmax multiclass head.

    Returns:
        Compiled `keras.Model` ready for training.

    Notes:
        Input shape: (batch, T=32, features=96).
        ReLU activations are prone to gradient explosion; gradient clipping must be
        applied in the optimizer (see training.train). This is a known gotcha.
    """
    units, output_activation = _head_units_and_activation(
        head=head, num_classes=num_classes
    )
    inputs, x = _input_layers(use_masking)
    x = layers.SimpleRNN(
        hidden_units,
        activation=activation,
        name="simple_rnn",
    )(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(units, activation=output_activation, name="classifier")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="rnn_sequence")


def build_lstm(
    activation: str = "tanh",
    use_masking: bool = False,
    head: str = "binary",
    hidden_units: int = 64,
    dropout: float = 0.2,
    num_classes: int | None = None,
) -> keras.Model:
    """
    Build an LSTM model for sequence classification.

    Args:
        activation: Recurrent activation ("tanh" or "relu").
        use_masking: If True, add a Masking layer before the LSTM.
        head: Output head ("binary" or "multiclass").
        hidden_units: Number of hidden units in the LSTM.
        dropout: Dropout rate.
        num_classes: Preferred output size. ``1`` gives a sigmoid binary head;
            values above one give a softmax multiclass head.

    Returns:
        Compiled `keras.Model`.

    Notes:
        Input shape: (batch, T=32, features=96).
        LSTM is more resistant to vanishing gradients than vanilla RNN.
    """
    units, output_activation = _head_units_and_activation(
        head=head, num_classes=num_classes
    )
    inputs, x = _input_layers(use_masking)
    x = layers.LSTM(
        hidden_units,
        activation=activation,
        name="lstm",
    )(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(units, activation=output_activation, name="classifier")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="lstm_sequence")


def build_bilstm(
    activation: str = "tanh",
    use_masking: bool = False,
    head: str = "binary",
    hidden_units: int = 64,
    dropout: float = 0.2,
    num_classes: int | None = None,
) -> keras.Model:
    """
    Build a Bidirectional LSTM model for sequence classification.

    Args:
        activation: Recurrent activation ("tanh" or "relu").
        use_masking: If True, add a Masking layer before the BiLSTM.
        head: Output head ("binary" or "multiclass").
        hidden_units: Number of hidden units in each LSTM direction.
        dropout: Dropout rate.
        num_classes: Preferred output size. ``1`` gives a sigmoid binary head;
            values above one give a softmax multiclass head.

    Returns:
        Compiled `keras.Model`.

    Notes:
        Input shape: (batch, T=32, features=96).
        BiLSTM processes the sequence in both directions.
    """
    units, output_activation = _head_units_and_activation(
        head=head, num_classes=num_classes
    )
    inputs, x = _input_layers(use_masking)
    x = layers.Bidirectional(
        layers.LSTM(hidden_units, activation=activation),
        name="bilstm",
    )(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(units, activation=output_activation, name="classifier")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="bilstm_sequence")
