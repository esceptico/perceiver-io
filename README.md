# Perceiver IO
Unofficial implementation of 
[Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795)


# Installation
**From PyPI**
```shell
pip install -U perceiver-io-pytorch
```


# Usage

```python
import torch

from perceiver_io.decoders import PerceiverDecoder
from perceiver_io.encoder import PerceiverEncoder
from perceiver_io import PerceiverIO

num_latents = 128
latent_dim = 256
input_dim = 64

decoder_query_dim = 4

encoder = PerceiverEncoder(
    num_latents=num_latents,
    latent_dim=latent_dim,
    input_dim=input_dim,
    num_self_attn_per_block=8,
    num_blocks=1
)
decoder = PerceiverDecoder(
    latent_dim=latent_dim,
    query_dim=decoder_query_dim
)
perceiver = PerceiverIO(encoder, decoder)

inputs = torch.randn(2, 16, input_dim)
output_query = torch.randn(2, 3, decoder_query_dim)

perceiver(inputs, output_query)  # shape = (2, 3, 4)

```

# List of implemented decoders
* ProjectionDecoder
* ClassificationDecoder
* PerceiverDecoder

# Example architectures:
* [Perceiver for LM](examples/language_modelling.py)

# Citation
```bibtex
@misc{jaegle2021perceiver,
    title   = {Perceiver IO: A General Architecture for Structured Inputs & Outputs},
    author  = {Andrew Jaegle and Sebastian Borgeaud and Jean-Baptiste Alayrac and Carl Doersch and Catalin Ionescu and David Ding and Skanda Koppula and Andrew Brock and Evan Shelhamer and Olivier Hénaff and Matthew M. Botvinick and Andrew Zisserman and Oriol Vinyals and João Carreira},
    year    = {2021},
    eprint  = {2107.14795},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
