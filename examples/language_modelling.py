from typing import Optional

import torch
from torch import nn

from src.perceiver.decoders import PerceiverDecoder
from src.perceiver.encoder import PerceiverEncoder
from src.perceiver.perceiver import PerceiverIO


class PerceiverLM(nn.Module):
    """Encoder-decoder based language model."""
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embedding_dim: int,
        num_latents: int = 256,
        latent_dim: int = 512,
        num_self_attn_heads=8,
        self_attn_head_dim=None,
        cross_attn_head_dim=None,
        self_attn_widening_factor=1,
        cross_attn_widening_factor=1,
        num_blocks=1,
        num_self_attn_per_block=12,
        dropout: float = 0.0
    ):
        """Constructor.

        Args:
            vocab_size: Size of vocabulary.
            max_seq_len: Maximum length of token sequence.
            embedding_dim: Dimension of token embedding.
            num_latents: Number of latent vectors. Defaults to 256.
            latent_dim: Dimension of latent vector. Defaults to 512.
            num_self_attn_heads: Number of self-attention heads. Defaults to 8.
            self_attn_head_dim: Size of self-attention head. If None,this
                value will be calculated as latent_dim / num_self_attn_heads.
                Defaults to None.
            cross_attn_head_dim: Size of cross-attention head. If None,this
                value will be equal latent_dims. Defaults to None.
            self_attn_widening_factor: Widening factor in self-attention
                feed-forward layer. Defaults to 1.
            cross_attn_widening_factor: Widening factor in cross-attention
                feed-forward layer. Defaults to 1.
            num_blocks: Number of transformer blocks. Defaults to 1.
            num_self_attn_per_block: Number of self-attention modules per
                transformer block. Defaults to 12.
            dropout: Dropout probability. Defaults to 0.
        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        encoder = PerceiverEncoder(
            num_latents=num_latents,
            latent_dim=latent_dim,
            input_dim=embedding_dim,
            num_self_attn_per_block=num_self_attn_per_block,
            num_blocks=num_blocks,
            cross_attn_head_dim=cross_attn_head_dim,
            self_attn_head_dim=self_attn_head_dim,
            num_self_attn_heads=num_self_attn_heads,
            cross_attn_widening_factor=cross_attn_widening_factor,
            self_attn_widening_factor=self_attn_widening_factor,
            dropout=dropout,
        )
        decoder = PerceiverDecoder(
            latent_dim=latent_dim,
            query_dim=embedding_dim,
            widening_factor=cross_attn_widening_factor,
            projection_dim=vocab_size
        )
        self.perceiver = PerceiverIO(encoder, decoder)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            inputs: Tensor of token ids.
            mask: Token mask. Mask values selected in [0, 1]. Defaults to None.

        Returns:
            Tensor of shape (batch_size, seq_len, vocab_size).
        """
        seq_len = inputs.size(1)
        token_embeddings = self.token_embedding(inputs)
        positions_ids = torch.arange(seq_len, device=inputs.device).view(1, -1)
        position_embeddings = self.position_embedding(positions_ids)
        embeddings = token_embeddings + position_embeddings

        outputs = self.perceiver(
            inputs=embeddings,
            query=position_embeddings,
            input_mask=mask,
            query_mask=mask
        )
        return outputs
