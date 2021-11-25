from typing import Optional
import torch
from torch import nn

from perceiver_io import PerceiverEncoder, PerceiverDecoder, PerceiverIO, ClassificationDecoder
from perceiver_io.adapter import ImageInputAdapter, ClassificationOutputAdapter
class PerceiverLM(nn.Module):
    """Encoder-decoder based language model."""
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embedding_dim: int,
        num_latents: int = 256,
        latent_dim: int = 1280,
        qk_out_dim = 8*32,
        v_out_dim = None,
        num_self_attn_heads=8,
        num_cross_attn_heads=8,
        num_decoder_attn_heads=8,
        self_attn_widening_factor=1,
        cross_attn_widening_factor=1,
        num_blocks=1,
        num_self_attn_per_block=12,
        dropout: float = 0.0
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.query_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.decoder_token_bias = nn.Parameter(torch.randn(vocab_size))
        if v_out_dim is None: v_out_dim = latent_dim
        encoder = PerceiverEncoder(
            num_latents=num_latents,
            latent_dim=latent_dim,
            input_dim=embedding_dim,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            num_self_attn_per_block=num_self_attn_per_block,
            num_blocks=num_blocks,
            num_self_attn_heads=num_self_attn_heads,
            num_cross_attn_heads=num_cross_attn_heads,
            cross_attn_widening_factor=cross_attn_widening_factor,
            self_attn_widening_factor=self_attn_widening_factor,
            dropout=dropout,
        )
        decoder = PerceiverDecoder(
            #num_classes=10,
            latent_dim=latent_dim,
            query_dim=embedding_dim,
            qk_out_dim=qk_out_dim,
            v_out_dim=embedding_dim,
            num_heads=num_decoder_attn_heads,
            widening_factor=cross_attn_widening_factor,
            projection_dim=None
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
        print("after transform size is", inputs.size())
        input_adapter = ImageInputAdapter(
            image_shape=torch.squeeze(inputs).shape,#image_shape assuming bs = 1 which hasto be for pretrained weights
            num_frequency_bands=256)#args.num_frequency_bands)
        image_adapter = input_adapter.forward(inputs)
        print("image adapter shape is ",image_adapter.shape)
        seq_len = image_adapter.shape[1]
        fst = torch.squeeze(image_adapter)
        linear0 = nn.Linear(image_adapter.shape[2],768)
        token_embeddings = linear0(fst)
        print("token embeddings shape is", token_embeddings.shape)
        positions_ids = torch.arange(seq_len, device=inputs.device).view(1, -1)
        position_embeddings = self.position_embedding(positions_ids)
        embeddings = token_embeddings + position_embeddings
        query_embeddings = self.query_embedding(positions_ids)
        outputs = self.perceiver(
            inputs=embeddings,
            query=query_embeddings,
            input_mask=mask,
            query_mask=mask
        )
        print(outputs.shape)
        logits = torch.matmul(outputs, self.token_embedding.weight.T) + self.decoder_token_bias
        print(logits.shape)
        last = logits.reshape(1,image_adapter.shape[1]*logits.shape[2]);
        print(last.shape)
        linear1 = nn.Linear(image_adapter.shape[1]*logits.shape[2],10)
        output = linear1(last)
        print(output.shape)
        return output
