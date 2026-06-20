"""Sequential models: vanilla RNN, LSTM, and Bi-LSTM with optional masking."""

from typing import Literal, Optional
import tensorflow as tf

keras = tf.keras
layers = tf.keras.layers


def build_rnn(
    activation: str = "tanh",
    use_masking: bool = False,
    head: str = "binary",
    hidden_units: int = 64,
    dropout: float = 0.2,
) -> keras.Model:
    """
    Build a vanilla RNN model for sequence classification.

    Args:
        activation: Recurrent activation ("tanh" or "relu").
        use_masking: If True, add a Masking layer before the RNN.
        head: Output head ("binary" → sigmoid+1, "multiclass" → softmax+10).
        hidden_units: Number of hidden units in the RNN.
        dropout: Dropout rate.

    Returns:
        Compiled `keras.Model` ready for training.

    Notes:
        Input shape: (batch, T=32, features=96).
        ReLU activations are prone to gradient explosion; gradient clipping must be
        applied in the optimizer (see training.train). This is a known gotcha.
    """
    # TODO: Implement vanilla RNN model.
    # 1. Input: (batch, 32, 96)
    # 2. Optional Masking(mask_value=0.0) if use_masking=True
    # 3. RNN layer with activation
    # 4. Dropout
    # 5. Classification head (sigmoid or softmax based on head)
    raise NotImplementedError("build_rnn not yet implemented")


def build_lstm(
    activation: str = "tanh",
    use_masking: bool = False,
    head: str = "binary",
    hidden_units: int = 64,
    dropout: float = 0.2,
) -> keras.Model:
    """
    Build an LSTM model for sequence classification.

    Args:
        activation: Recurrent activation ("tanh" or "relu").
        use_masking: If True, add a Masking layer before the LSTM.
        head: Output head ("binary" or "multiclass").
        hidden_units: Number of hidden units in the LSTM.
        dropout: Dropout rate.

    Returns:
        Compiled `keras.Model`.

    Notes:
        Input shape: (batch, T=32, features=96).
        LSTM is more resistant to vanishing gradients than vanilla RNN.
    """
    # TODO: Implement LSTM model.
    # Same structure as build_rnn but with LSTM layer.
    raise NotImplementedError("build_lstm not yet implemented")


def build_bilstm(
    activation: str = "tanh",
    use_masking: bool = False,
    head: str = "binary",
    hidden_units: int = 64,
    dropout: float = 0.2,
) -> keras.Model:
    """
    Build a Bidirectional LSTM model for sequence classification.

    Args:
        activation: Recurrent activation ("tanh" or "relu").
        use_masking: If True, add a Masking layer before the BiLSTM.
        head: Output head ("binary" or "multiclass").
        hidden_units: Number of hidden units in each LSTM direction.
        dropout: Dropout rate.

    Returns:
        Compiled `keras.Model`.

    Notes:
        Input shape: (batch, T=32, features=96).
        BiLSTM processes the sequence in both directions.
    """
    # TODO: Implement Bidirectional LSTM model.
    # Same structure as build_lstm but wrapped in Bidirectional.
    raise NotImplementedError("build_bilstm not yet implemented")
