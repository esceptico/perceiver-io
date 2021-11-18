from abc import ABCMeta, abstractmethod
from typing import Optional

import torch
from torch import nn

from perceiver_io.attention import CrossAttention


class BasePerceiverDecoder(nn.Module, metaclass=ABCMeta):
    """Abstract decoder class."""
    @abstractmethod
    def forward(
        self,
        *,
        query: torch.Tensor,
        latents: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None
    ):
        return NotImplementedError


class ProjectionDecoder(BasePerceiverDecoder):
    """Projection decoder without using a cross-attention layer."""
    def __init__(self, latent_dim: int, num_classes: int):
        super().__init__()
        self.projection = nn.Linear(latent_dim, num_classes)

    def forward(
        self,
        *,
        query: torch.Tensor,
        latents: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None
    ):
        latents = latents.mean(dim=1)
        logits = self.projection(latents)
        return logits


class PerceiverDecoder(BasePerceiverDecoder):
    """Basic cross-attention decoder."""
    def __init__(
        self,
        latent_dim: int,
        query_dim: int,
        widening_factor: int = 1,
        num_heads: int = 1,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        projection_dim: Optional[int] = None,
        use_query_residual: bool = False
    ):
        super().__init__()
        self.cross_attention = CrossAttention(
            kv_dim=latent_dim,
            q_dim=query_dim,
            widening_factor=widening_factor,
            num_heads=num_heads,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            use_query_residual=use_query_residual
        )
        if projection_dim is not None:
            self.projection = nn.Linear(query_dim, projection_dim)
        else:
            self.projection = nn.Identity()

    def forward(
        self,
        *,
        query: torch.Tensor,
        latents: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None
    ):
        if q_mask is not None:
            q_mask = q_mask[:, None, None, :].transpose(-2, -1)
        outputs = self.cross_attention(
            inputs_kv=latents,
            inputs_q=query,
            attention_mask=q_mask
        )
        return self.projection(outputs)


class ClassificationDecoder(BasePerceiverDecoder):
    """Classification decoder. Based on PerceiverDecoder."""
    def __init__(
        self,
        num_classes: int,
        latent_dim: int,
        widening_factor: int = 1,
        num_heads: int = 1,
        head_dim: Optional[int] = None
    ):
        super().__init__()
        self.task_ids = nn.Parameter(torch.randn(1, num_classes))
        self.decoder = PerceiverDecoder(
            latent_dim=latent_dim,
            query_dim=num_classes,
            widening_factor=widening_factor,
            num_heads=num_heads,
            head_dim=head_dim,
            projection_dim=None,
            use_query_residual=False
        )

    def forward(
        self,
        *,
        query: torch.Tensor,
        latents: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None
    ):
        batch_size = latents.size(0)
        logits = self.decoder.forward(
            query=self.task_ids.repeat(batch_size, 1, 1),
            latents=latents,
            q_mask=q_mask
        )
        return logits.squeeze(1)

