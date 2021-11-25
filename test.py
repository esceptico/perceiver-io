import pickle
with open("deepmind_assets/language_perceiver_io_bytes.pickle", "rb") as f:
    params = pickle.loads(f.read())
#import argparse
#import pytorch_lightning as pl
from perceiver_io.perceiver_im import PerceiverLM
#from data import MNISTDataModule
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


transform_train = transforms.Compose([
transforms.RandomCrop(32, padding=4),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
NUM_CLASSES=10
trainset = torchvision.datasets.STL10(root='./data',  split='train',
                                          download=False, transform=transform_train)
testset = torchvision.datasets.STL10(root='./data', split='test',
                                          download=False, transform=transform_train)
trainset, validset = torch.utils.data.random_split(trainset, 
                                                      [int(len(trainset)*0.8),len(trainset)- 
                                                      int(len(trainset)*0.8)])
#parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser = pl.Trainer.add_argparse_args(parser)
#parser = MNISTDataModule.setup_parser(parser)
#parser = pl.Trainer.add_argparse_args(parser)
#data_module = MNISTDataModule.create(args)
#group = parser.add_argument_group('main')
#group.add_argument('--experiment', default='img_clf', help=' ')
# Ignored at the moment, dataset is hard-coded ...
#group.add_argument('--dataset', default='mnist', choices=['mnist'], help=' ')
#args = parser.parse_args()
#data_module = MNISTDataModule.create(args)
dims = (3, 32, 32)
batch_size = 1
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                            shuffle=False,num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)
model = PerceiverLM(image_shape=dims,
                    num_classes=NUM_CLASSES,
                    num_frequency_bands=262, #not sure what num frequence band should be vocab_size  or max_seq_len or 32
                    vocab_size=262, 
                    max_seq_len=2048, 
                    embedding_dim=768, 
                    num_latents=768, 
                    latent_dim=1280,#NUM_CLASSES, 
                    qk_out_dim=256, 
                    num_self_attn_per_block=26)

state_dict = {}
model_enc_base = 'perceiver.encoder.'
params_enc_base = 'perceiver_encoder/~/'

#state_dict['token_embedding.weight'] = params['embed']['embeddings']
#state_dict['decoder_token_bias'] = params['embedding_decoder']['bias']
#state_dict['position_embedding.weight'] = params['trainable_position_encoding']['pos_embs']
#state_dict['query_embedding.weight'] = params['basic_decoder/~/trainable_position_encoding']['pos_embs']
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

model.eval()
#one way
test_loss = 0
total_images = 0
correct_images = 0
net.eval()
with torch.no_grad():
    for batch_index, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        outputs = model.forward(torch.tensor(images))
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total_images += labels.size(0)
        correct_images += predicted.eq(labels).sum().item()
        print(batch_index, len(testloader), 'Loss: %.3f | Accuracy: %.3f%% (%d/%d)'
                % (test_loss/(batch_index+1), 100.*correct_images/total_images, correct_images, total_images))
        test_accuracy = 100.*correct_images/total_images
print("accuracy of test set is",test_accuracy)
#return test_loss/(batch_index+1)
#out = model.forward(torch.tensor(testloader))

#other way
# plugins = pl.plugins.DDPPlugin(find_unused_parameters=False)
# logger = pl.loggers.TensorBoardLogger("logs", name=args.experiment)
# callbacks = [model_checkpoint_callback(save_top_k=1)]
# args = parser.parse_args()
# if args.one_cycle_lr:
#     callbacks.append(learning_rate_monitor_callback())

# trainer = pl.Trainer.from_argparse_args(args, plugins=plugins, callbacks=callbacks, logger=logger)
# trainer.fit(model, data_module)

