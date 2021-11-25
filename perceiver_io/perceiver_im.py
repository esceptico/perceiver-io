from typing import Optional
import torch
from torch import nn

from perceiver_io import PerceiverEncoder, PerceiverDecoder, PerceiverIO
from perceiver_io.adapter import ImageInputAdapter, ClassificationOutputAdapter

class PerceiverLM(nn.Module):
    """Encoder-decoder based language model."""
    def __init__(
        self,
        image_shape: int,
        num_classes: int,
        num_frequency_bands: int,
        vocab_size: int,
        max_seq_len: int,
        embedding_dim: int,
        num_latents: int = 768,
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
        input_adapter = ImageInputAdapter(
            image_shape=image_shape,
            num_frequency_bands=num_frequency_bands)
        embedding_dim2 = input_adapter.num_input_channels #doubtful here
        print("embedding dim image input is", embedding_dim2)
        #for now dont change v_out_dim and q_out_dim
        encoder = PerceiverEncoder(
            num_latents=256,#num_latents,
            latent_dim=latent_dim,
            input_dim=embedding_dim2,
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
        output_adapter = ClassificationOutputAdapter(
            num_classes=num_classes,
            num_output_channels=768) #latent_dim) #not sure on num_output_channel
        embedding_dim1= output_adapter.output_shape
        decoder = PerceiverDecoder(
            latent_dim=768, #latent_dim,
            query_dim=embedding_dim1[-1],
            qk_out_dim=768 ,#qk_out_dim,
            #v_out_dim=embedding_dim1,
            num_heads=num_decoder_attn_heads,
            widening_factor=cross_attn_widening_factor,
            #projection_dim=768#None
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
        # seq_len = inputs.size(1)
        # pos = input_adapter._positions()
        # enc = input_adapter._position_encodings(pos)
        # enc = rearrange(enc, '... c -> (...) c')
        # position_embeddings = enc
        # query_embeddings = enc
        # b = 1 #for now one batch size
        # x = input_adapter.image_shape
        # x_enc = repeat(position_embeddings, '... -> b ...', b=b)
        # x = rearrange(x, 'b ... c -> b (...) c')
        # token_embedding = x
        # embeddings = torch.cat([x, x_enc], dim=-1)
        input_adapter = ImageInputAdapter(
            image_shape=torch.tensor(inputs).size(),#image_shape
            num_frequency_bands=262)#args.num_frequency_bands)
        print(torch.tensor(input_adapter))
        token_embeddings = self.token_embedding(input_adapter)
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
        #not sure if this is token embedding
        # logits = torch.matmul(outputs, self.token_embedding.weight.T)
        # decoder_token_bias = nn.Parameter(torch.randn(logits.size))
        # logits = logits + decoder_token_bias
        #apply softmax layer
        logits = torch.matmul(outputs, self.token_embedding.weight.T) + self.decoder_token_bias
        return logits
