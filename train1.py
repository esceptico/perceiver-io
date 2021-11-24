import pickle
with open("deepmind_assets/language_perceiver_io_bytes.pickle", "rb") as f:
    params = pickle.loads(f.read())
from perceiver_io.perceiver_lm import PerceiverLM
import torch.nn as nn
import torch

model = PerceiverLM(vocab_size=262, max_seq_len=2048, embedding_dim=768, num_latents=256, latent_dim=1280,qk_out_dim=256, num_self_attn_per_block=26)
state_dict = {}
model_enc_base = 'perceiver.encoder.'
params_enc_base = 'perceiver_encoder/~/'

state_dict['token_embedding.weight'] = params['embed']['embeddings']
state_dict['decoder_token_bias'] = params['embedding_decoder']['bias']
state_dict['position_embedding.weight'] = params['trainable_position_encoding']['pos_embs']
state_dict['query_embedding.weight'] = params['basic_decoder/~/trainable_position_encoding']['pos_embs']
state_dict[f'{model_enc_base}latents'] = params[f'{params_enc_base}trainable_position_encoding']['pos_embs']

def copy_attention_params(model_base, params_base):
    global state_dict
    state_dict[f'{model_base}attention.q.weight'] = params[f'{params_base}attention/linear']['w'].T
    state_dict[f'{model_base}attention.q.bias'] = params[f'{params_base}attention/linear']['b']
    state_dict[f'{model_base}attention.k.weight'] = params[f'{params_base}attention/linear_1']['w'].T
    state_dict[f'{model_base}attention.k.bias'] = params[f'{params_base}attention/linear_1']['b']
    state_dict[f'{model_base}attention.v.weight'] = params[f'{params_base}attention/linear_2']['w'].T
    state_dict[f'{model_base}attention.v.bias'] = params[f'{params_base}attention/linear_2']['b']
    state_dict[f'{model_base}attention.projection.weight'] = params[f'{params_base}attention/linear_3']['w'].T
    state_dict[f'{model_base}attention.projection.bias'] = params[f'{params_base}attention/linear_3']['b']

    if 'self_attention' in params_base:
        state_dict[f'{model_base}layer_norm.weight'] = params[f'{params_base}layer_norm']['scale']
        state_dict[f'{model_base}layer_norm.bias'] = params[f'{params_base}layer_norm']['offset']
        state_dict[f'{model_base}qkv_layer_norm.weight'] = params[f'{params_base}layer_norm_1']['scale']
        state_dict[f'{model_base}qkv_layer_norm.bias'] = params[f'{params_base}layer_norm_1']['offset']
    else:
        state_dict[f'{model_base}q_layer_norm.weight'] = params[f'{params_base}layer_norm']['scale']
        state_dict[f'{model_base}q_layer_norm.bias'] = params[f'{params_base}layer_norm']['offset']
        state_dict[f'{model_base}kv_layer_norm.weight'] = params[f'{params_base}layer_norm_1']['scale']
        state_dict[f'{model_base}kv_layer_norm.bias'] = params[f'{params_base}layer_norm_1']['offset']
        state_dict[f'{model_base}qkv_layer_norm.weight'] = params[f'{params_base}layer_norm_2']['scale']
        state_dict[f'{model_base}qkv_layer_norm.bias'] = params[f'{params_base}layer_norm_2']['offset']

    state_dict[f'{model_base}mlp.mlp.0.weight'] = params[f'{params_base}mlp/linear']['w'].T
    state_dict[f'{model_base}mlp.mlp.0.bias'] = params[f'{params_base}mlp/linear']['b']
    state_dict[f'{model_base}mlp.mlp.2.weight'] = params[f'{params_base}mlp/linear_1']['w'].T
    state_dict[f'{model_base}mlp.mlp.2.bias'] = params[f'{params_base}mlp/linear_1']['b']

copy_attention_params(f'{model_enc_base}cross_attn.', f'{params_enc_base}cross_attention/')
copy_attention_params(f'perceiver.decoder.cross_attention.', f'basic_decoder/cross_attention/')

for i in range(26):
    copy_attention_params(f'{model_enc_base}self_attention_block.{i}.', f'{params_enc_base}self_attention{"_%d"%i if i else ""}/')
state_dict = {k: torch.tensor(v) for k,v in state_dict.items()}
model.load_state_dict(state_dict)

from deepmind_assets import bytes_tokenizer
import numpy as np
MAX_SEQ_LEN = 2048
# The tokenizer is just UTF-8 encoding (with an offset)
tokenizer = bytes_tokenizer.BytesTokenizer()
input_str = "This is an incomplete sentence where some words are missing."
input_tokens = tokenizer.to_int(input_str)

# Mask " missing.". Note that the model performs much better if the masked chunk
# starts with a space.
input_tokens[51:60] = tokenizer.mask_token
print("Tokenized string without masked bytes:")
print(tokenizer.to_string(input_tokens))

#@title Pad and reshape inputs
inputs = input_tokens[None]
input_mask = np.ones_like(inputs)

def pad(max_sequence_length: int, inputs, input_mask):
    input_len = inputs.shape[1]
    assert input_len <= max_sequence_length
    pad_len = max_sequence_length - input_len
    padded_inputs = np.pad(
      inputs,
      pad_width=((0, 0), (0, pad_len)),
      constant_values=tokenizer.pad_token)
    padded_mask = np.pad(
      input_mask,
      pad_width=((0, 0), (0, pad_len)),
      constant_values=0)
    return padded_inputs, padded_mask

inputs, input_mask = pad(MAX_SEQ_LEN, inputs, input_mask)

model.eval()
out = model.forward(torch.tensor(inputs), torch.tensor(input_mask))
masked_tokens_predictions = out[0, 51:60].argmax(dim=-1)
print("Greedy predictions:")
print(masked_tokens_predictions)
print()
print("Predicted string:")
print(tokenizer.to_string(masked_tokens_predictions.cpu().detach().numpy()))


