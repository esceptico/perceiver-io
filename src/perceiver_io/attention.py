from typing import Optional

import torch
from einops import rearrange
from torch import nn


class MultiHeadAttention(nn.Module):
    """Multi-head attention"""
    def __init__(
        self,
        kv_dim: int,
        q_dim: int,
        *,
        head_dim: Optional[int] = None,
        num_heads: int = 1,
        dropout: float = 0.0
    ):
        """Constructor.

        Args:
            kv_dim: Size of input key and value vectors.
            q_dim: Size of input query vector.
            head_dim: Size of head. If None will be equal to
                `q_dim / num_heads`. Defaults to None.
            num_heads: Number of heads. Defaults to 1.
            dropout: Dropout probability. Defaults to 0.0.
        """
        super().__init__()
        hidden_dim = q_dim
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads

        if head_dim is not None:
            hidden_dim = head_dim * num_heads
            self.head_dim = head_dim

        self.k = nn.Linear(kv_dim, hidden_dim)
        self.q = nn.Linear(q_dim, hidden_dim)
        self.v = nn.Linear(kv_dim, hidden_dim)
        self.projection = nn.Linear(hidden_dim, q_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        inputs_kv: torch.Tensor,
        inputs_q: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            inputs_kv: Key/Value embeddings of shape (B, M, C).
            inputs_q: Query embeddings of shape (B, N, D)
            attention_mask: Tensor of shape (B, N, M).

        Returns:
            Tensor of shape (B, N, D)
        """
        k, q, v = self.k(inputs_kv), self.q(inputs_q), self.v(inputs_kv)
        k = rearrange(k, 'b s (n h) -> b n s h', h=self.head_dim)
        q = rearrange(q, 'b s (n h) -> b n s h', h=self.head_dim)
        v = rearrange(v, 'b s (n h) -> b n s h', h=self.head_dim)
        attention = (q @ k.transpose(-2, -1) / self.scale)
        if attention_mask is not None:
            min_value = torch.finfo(attention.dtype).min
            extended_mask = (1 - attention_mask) * min_value
            attention = attention + extended_mask
        attention = attention.softmax(dim=-1)
        attention = self.dropout(attention)
        if attention_mask is not None:
            attention = attention.masked_fill(1 - attention_mask, value=0)
        weighted = rearrange(attention @ v, 'b n s h -> b s (n h)')
        return self.projection(weighted)


class FeedForward(nn.Module):
    """Transformer Feed-Forward network."""
    def __init__(
        self,
        dim: int,
        widening_factor: int = 4,
        dropout: float = 0.0
    ):
        """Constructor.

        Args:
            dim: Dimension of input tensor.
            widening_factor: Widening factor. Defaults to 4.
            dropout: Dropout probability. Defaults to 0.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * widening_factor),
            nn.GELU(),
            nn.Linear(dim * widening_factor, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


class SelfAttention(nn.Module):
    """Self-attention module."""
    def __init__(
        self,
        *,
        hidden_dim: int,
        widening_factor: int = 4,
        num_heads: int = 1,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0
    ):
        """Constructor.

        Args:
            hidden_dim: Dimension of input tensor.
            widening_factor: Feed-forward network widening factor.
                Defaults to 4.
            num_heads: Number of attention heads. Defaults to 1.
            head_dim: Size of single attention head. If None,this
                value will be calculated as hidden_dim / num_heads.
                Defaults to None.
            dropout: Dropout probability. Defaults to 0.
            attention_dropout: Attention scores probability. Defaults to 0.
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.qkv_layer_norm = nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadAttention(
            kv_dim=hidden_dim,
            q_dim=hidden_dim,
            head_dim=head_dim,
            num_heads=num_heads,
            dropout=attention_dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.mlp = FeedForward(hidden_dim, widening_factor, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x: Input tensor of shape (B, M, C).
            attention_mask: Input mask tensor of shape (B, M, M).
                Mask values selected in [0, 1]. Defaults to None.
        """
        x_norm = self.layer_norm(x)
        attention = self.attention(
            inputs_kv=x_norm,
            inputs_q=x_norm,
            attention_mask=attention_mask
        )
        attention = self.dropout(attention)
        x = x + attention
        x = x + self.mlp(self.qkv_layer_norm(x))
        return x


class CrossAttention(nn.Module):
    """Cross-attention module."""
    def __init__(
        self,
        *,
        kv_dim: int,
        q_dim: int,
        widening_factor: int = 1,
        num_heads: int = 1,
        head_dim: Optional[int] = None,
        use_query_residual: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0
    ):
        """Constructor.

        Args:
            kv_dim: Dimension of key/value input tensor.
            q_dim: Dimension of query input tensor.
            widening_factor: Feed-forward network widening factor.
                Defaults to 4.
            num_heads: Number of attention heads. Defaults to 1.
            head_dim: Size of single attention head. If None,this
                value will be calculated as hidden_dim / num_heads.
                Defaults to None.
            use_query_residual: Indicates whether to use query residual in
                cross-attention. Defaults to True.
            dropout: Dropout probability. Defaults to 0.
            attention_dropout: Attention scores probability. Defaults to 0.
        """
        super().__init__()
        self.use_query_residual = use_query_residual
        self.kv_layer_norm = nn.LayerNorm(kv_dim)
        self.q_layer_norm = nn.LayerNorm(q_dim)
        self.qkv_layer_norm = nn.LayerNorm(q_dim)
        self.attention = MultiHeadAttention(
            kv_dim=kv_dim,
            q_dim=q_dim,
            head_dim=head_dim,
            num_heads=num_heads,
            dropout=attention_dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.mlp = FeedForward(q_dim, widening_factor, dropout)

    def forward(
        self,
        inputs_kv: torch.Tensor,
        inputs_q: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            inputs_kv: Key/Value embeddings of shape (B, M, C).
            inputs_q: Query embeddings of shape (B, N, D)
            attention_mask: Tensor of shape (B, N, M). Mask values selected
                in [0, 1]. Defaults to None.
        """
        attention = self.attention(
            inputs_kv=self.kv_layer_norm(inputs_kv),
            inputs_q=self.q_layer_norm(inputs_q),
            attention_mask=attention_mask
        )
        attention = self.dropout(attention)
        if self.use_query_residual:
            x = inputs_q + attention
        else:
            x = attention
        x = x + self.mlp(self.qkv_layer_norm(x))
        return x
