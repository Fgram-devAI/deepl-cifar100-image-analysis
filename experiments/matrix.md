# Experiment matrix definition

ARCHITECTURES = ["rnn", "lstm", "bilstm", "vit", "transfer"]

ACTIVATIONS = ["tanh", "relu"]  # For RNN family; N/A for ViT and Transfer

HEADS = ["binary", "multiclass"]

MASKING_VARIANTS = [False, True]  # off, on

# Matrix summary :
# 5 architectures × 2 activations (RNN family) × 2 heads × 2 masking
# = ~20–30 core configurations (excluding ViT/Transfer which don't have activation axis)

MATRIX = """
| Architecture | Hidden activation | Output head | Masking | Notes |
|---|---|---|---|---|
| Vanilla RNN | tanh / relu | binary / multiclass | off / on | Baseline sequential; prone to vanishing/exploding gradients |
| LSTM | tanh / relu | binary / multiclass | off / on | Improved gradient flow vs RNN |
| Bi-LSTM | tanh / relu | binary / multiclass | off / on | Bidirectional context |
| ViT | N/A | binary / multiclass | N/A | From-scratch tiny transformer; no activation/masking axes |
| MobileNetV3 (Transfer) | N/A | binary / multiclass | N/A | CNN upper benchmark; frozen backbone |

**Final:** OOD eval on Midjourney v6 for best binary models.
"""

if __name__ == "__main__":
    print(MATRIX)
