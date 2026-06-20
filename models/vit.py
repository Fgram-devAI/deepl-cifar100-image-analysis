"""Vision Transformer: from-scratch tiny ViT implementation."""

from typing import Literal
import tensorflow as tf

keras = tf.keras
layers = tf.keras.layers


class PatchEmbedding(layers.Layer):
    """
    Extract patches from images and embed them.

    Converts (batch, H, W, C) images to (batch, num_patches, embedding_dim).
    """

    def __init__(self, patch_size: int, embedding_dim: int, **kwargs):
        """
        Args:
            patch_size: Size of each patch (e.g., 4 for 4×4 patches).
            embedding_dim: Dimension of patch embeddings.
        """
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        # TODO: Implement patch extraction and linear projection.
        # Use Conv2D with kernel=patch_size, strides=patch_size, filters=embedding_dim.

    def call(self, images):
        """
        Args:
            images: (batch, H, W, C)
        Returns:
            (batch, num_patches, embedding_dim)
        """
        # TODO: Implement.
        raise NotImplementedError("PatchEmbedding.call not yet implemented")


class Attention(layers.Layer):
    """Single-head self-attention (required by project spec)."""

    def __init__(self, dim: int, **kwargs):
        """
        Args:
            dim: Embedding dimension.
        """
        super().__init__(**kwargs)
        self.dim = dim
        # TODO: Implement single-head attention layer.

    def call(self, x):
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            (batch, seq_len, dim)
        """
        # TODO: Implement scaled dot-product attention.
        raise NotImplementedError("Attention.call not yet implemented")


class TransformerBlock(layers.Layer):
    """Single transformer encoder block: attention + feedforward."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, **kwargs):
        """
        Args:
            dim: Embedding dimension.
            num_heads: Number of attention heads (typically 4–8 for small models).
            mlp_dim: Feedforward (MLP) hidden dimension.
        """
        super().__init__(**kwargs)
        self.dim = dim
        # TODO: Implement transformer block.
        # - Layer norm + MultiHeadAttention
        # - Layer norm + Dense feedforward (2 layers with activation)
        # - Residual connections

    def call(self, x):
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            (batch, seq_len, dim)
        """
        # TODO: Implement.
        raise NotImplementedError("TransformerBlock.call not yet implemented")


def build_vit(
    image_size: int = 32,
    patch_size: int = 4,
    embedding_dim: int = 128,
    num_blocks: int = 2,
    num_heads: int = 4,
    mlp_dim: int = 256,
    num_classes: int = 2,
    head: str = "binary",
) -> keras.Model:
    """
    Build a tiny from-scratch Vision Transformer.

    Args:
        image_size: Input image size (32×32 for CIFAKE).
        patch_size: Size of each patch (e.g., 4 → 64 patches for 32×32 image).
        embedding_dim: Embedding dimension (keep ≤128 for T4 memory).
        num_blocks: Number of transformer encoder blocks (≤2 for T4).
        num_heads: Number of attention heads in MultiHeadAttention.
        mlp_dim: Feedforward hidden dimension in transformer blocks.
        num_classes: Number of output classes (2 for binary, 10 for multiclass).
        head: Output head type ("binary" or "multiclass").

    Returns:
        Compiled `keras.Model`.

    Notes:
        Input shape: (batch, 32, 32, 3) — the image view (not row-as-timestep).
        Architecture:
          1. Patch embedding (32×32×3 → 64 patches of 128-dim)
          2. Positional embeddings
          3. num_blocks transformer encoder blocks
          4. Classification head (sigmoid or softmax)
        Keep this tiny to fit on a T4; avoid large models.
    """
    # TODO: Implement ViT model.
    # 1. Input: (batch, 32, 32, 3)
    # 2. PatchEmbedding to (batch, num_patches, embedding_dim)
    # 3. Add positional embeddings
    # 4. Stack num_blocks TransformerBlock layers
    # 5. Global average pooling or [CLS] token → classification head
    # 6. Sigmoid or softmax based on head
    raise NotImplementedError("build_vit not yet implemented")
