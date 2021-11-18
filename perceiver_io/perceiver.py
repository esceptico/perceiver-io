from typing import Optional

import torch
from torch import nn

from perceiver_io.decoders import BasePerceiverDecoder
from perceiver_io.encoder import PerceiverEncoder


class PerceiverIO(nn.Module):
    """Perceiver IO encoder-decoder architecture."""
    def __init__(
        self,
        encoder: PerceiverEncoder,
        decoder: BasePerceiverDecoder
    ):
        """Constructor.

        Args:
            encoder: Instance of Perceiver IO encoder.
            decoder: Instance of Perceiver IO decoder.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        inputs: torch.Tensor,
        query: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            inputs: Input tensor.
            query: Decoder query tensor. Can be a trainable or hand-made.
                Defaults to None.
            input_mask: Input mask tensor. Mask values selected in [0, 1].
                Defaults to None.
            query_mask: Decoder query mask tensor. Mask values selected in
                [0, 1]. Defaults to None.

        Returns:
            Output tensor.
        """
        latents = self.encoder(inputs, kv_mask=input_mask)
        outputs = self.decoder(
            query=query,
            latents=latents,
            q_mask=query_mask
        )
        return outputs
