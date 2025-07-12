
# --- Test UNET ---
# from models.mnist import UNET
# x = torch.rand([4, 3, 32, 32])
# unet = UNET(
#     in_channels=x.shape[1],
#     conv_channels=[64, 128, 256])
# y = unet(x)
#
# print(y.shape)


# Legacy Model Parameters
# model = ContinousThoughtMachine(
#     n_ticks = 32,
#     d_model = 256,
#     d_memory = 15,
#     n_sync_out = 64,
#     n_sync_action = 64,
#     d_input = 32,
#     n_heads = 8,
#     d_output = 10, # for both MNIST & CIFAR10
#     backbone = BackBone,
#     neuron_lvl_model = NLM,
#     synapse_model = Synapses,
#     dropout = 0,
#     neuron_selection_type = 'first-last',
#     n_random_pairing_self = 0,
# )


# --- Visualize ---
import json
import torch
import argparse
from models.ctm_dyn_kv import ContinousThoughtMachineDyn
from utils import prepare_data
from visual_utils import visualize_attention_map_with_dynamic_images, visualize_attention_map_with_images
from models.ctm import ContinousThoughtMachine
from models.mnist import Synapses, BackBone, NLM


parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, required=True, help="path of the run")
parser.add_argument("--num_images", type=int, help="number of images to run visualization on", default=1)
parser.add_argument("--dataset", type=str, help="Datset to visualize this run on")
args = parser.parse_args()

# Load the config
with open(f'{args.run}/config.json', 'r') as f:
    config = json.load(f)
training_args = config['training_args']
backbone_args = config['backbone_config']
synapse_args = config['synapses_config']
nlm_args = config['nlm_config']
model_args = config['model_config']

# Getting the Dataset to test this run on, Ex. when running the MNIST_COUNT model on MNIST classifier
dataset_name = training_args['name'] if args.dataset is None else args.dataset

if dataset_name == 'MNIST':
    label_text = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
elif dataset_name == 'CIFAR10':
    label_text = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
elif dataset_name == 'MNIST_COUNT':
    label_text = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
else:
    raise ValueError(f"No labels-text for {dataset_name}")

# Load the dataset
_, testloader = prepare_data(dataset_name, batch_size=args.num_images, n_ticks=model_args['n_ticks'])
image, label = next(iter(testloader))

# Load the Models
backbone = BackBone(d_input=backbone_args['d_input'], use_unet=backbone_args['use_unet'])

synapses = Synapses(d_model=synapse_args['d_model'])

nlm = NLM(
    d_memory=nlm_args['d_memory'],
    d_model=nlm_args['d_model'],
    memory_hidden_dims=nlm_args['memory_hidden_dims']
)
if training_args['name'] == 'MNIST_COUNT':
    ctm_type = ContinousThoughtMachineDyn
else:
    ctm_type = ContinousThoughtMachine

model = ctm_type(
    n_ticks = model_args['n_ticks'],
    d_model = model_args['d_model'],
    d_memory = model_args['d_memory'],
    n_sync_out = model_args['n_sync_out'],
    n_sync_action = model_args['n_sync_action'],
    d_input = model_args['d_input'],
    n_heads = model_args['n_heads'],
    d_output = model_args['d_output'], # for both MNIST & CIFAR10
    backbone = backbone,
    neuron_lvl_model = nlm,
    synapse_model = synapses,
    dropout = model_args['dropout'],
    neuron_selection_type = model_args['neuron_selection_type'],
    n_random_pairing_self = model_args['n_random_pairing_self'],
)
with open(f'{args.run}/run_summary.json', 'r') as f:
    paths = json.load(f)
model_weights_path = paths['files_saved']['model_weights']
state_dict = torch.load(model_weights_path)
model.load_state_dict(state_dict)

device = "cuda"
if dataset_name == 'MNIST_COUNT':
    visualize_attention_map_with_dynamic_images(dataset_name, model, image, label, label_text, device, path=f'{args.run}/viz')
else:
    visualize_attention_map_with_images(dataset_name, model, image, label, label_text, device, path=f'{args.run}/viz')

