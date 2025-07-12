import math
from typing import Optional
import torch
import numpy as np
import torch.nn as nn

from utils import compute_normalized_entropy
from models.mnist import BackBone, Synapses, NLM
class ContinousThoughtMachineDyn(nn.Module):
    def __init__(self, 
        n_ticks: int,                                   # total number of ticks
        d_model: int,                                   # width of model / number of 'neurons'
        d_memory: int,                                  # window length for the pre-activations of each neuron
        n_sync_out: int,                                # number of neurons used for output synchronization
        n_sync_action: int,                             # number of neurons used for computing attention queries
        d_input: int,                                   # Input and attention embd dimension
        n_heads: int,                                   # number of heads for the attention
        d_output: int,                                  # output dimension
        
        backbone: BackBone,
        neuron_lvl_model: NLM,
        synapse_model: Synapses,
        
        dropout: float = 0,
        # synapse_depth: int = 2,                       # UNET if > q else MLP
        # dropout_nlm: Optional[float] = None,
        # is_deep_nlms: bool = True,                    # Use deep NLMs as a default
        # is_layernorm_nlm: bool = True,
        # hidden_dims_nlm: Optional[int] = 32,          # number of hidden dimensions if deep-nlms 
        neuron_selection_type: str = 'random-pairing',
        n_random_pairing_self: int = 0                  # Number of neurons to select for self-to-self synch when random-pairing is used.
    ):
        super(ContinousThoughtMachineDyn, self).__init__()
        # --- Core Parameters ---

        # Model Neuron Parameters 
        self.n_ticks = n_ticks
        self.d_model = d_model
        self.d_memory = d_memory

        # Synchroniation Params
        self.n_sync_out = n_sync_out
        self.n_sync_action = n_sync_action

        # Input and Attention Params
        self.d_input = d_input
        self.n_heads = n_heads

        # --- Input and Attention ---
        self.backbone: BackBone = self.get_backbone(backbone)

        # Initializing the Q and KV projectors, and attention Module
        self.q_proj = nn.LazyLinear(self.d_input) # Initializes a weight matrix with [... , d_input] & works on the Synchronized Embeddings
        self.kv_proj = nn.Sequential(nn.LazyLinear(self.d_input),nn.LayerNorm(self.d_input))
        self.attention = nn.MultiheadAttention( # splits the n_input dim into n_head, and makes n_heads number of Q with dim n_input/n_head, applies multihead attention and then combines the final ouput and projects into desired dimension.
            embed_dim=self.d_input,
            num_heads=self.n_heads,
            dropout=dropout,
            batch_first=True # (batch, seq, feature) instead of (seq, batch, feature)
        ) # This module uses W_q, W_k, W_v internally and projects to embed_dim Check the forward() method for more!

        # --- Core CTM Modules ---
        self.synapses: Synapses = self.get_synapses(synapse_model)
        self.history_processor: NLM = self.get_neuron_lvl_models(neuron_lvl_model)

        # --- Init Pre-activations & their history/trace as params ---
        # variance-scaled uniform initialization!! A practical methods to keep gradients stable
        self.register_parameter('initial_activated_state', nn.Parameter(torch.zeros((d_model)).uniform_(-math.sqrt(1/(d_model)), math.sqrt(1/(d_model))), requires_grad=True))
        self.register_parameter('initial_activation_history', nn.Parameter(torch.zeros((d_model, d_memory)).uniform_(-math.sqrt(1/(d_model+d_memory)), math.sqrt(1/(d_model+d_memory))), requires_grad=True))

        # --- Synchronization ---

        self.neuron_select_type = neuron_selection_type
        
        # Calculate the repr size based on 'neuron_selection_type' & 'sync_type'
        self.sync_representation_size_action = self.calculate_sync_representation_size(self.n_sync_action)
        self.sync_representation_size_out = self.calculate_sync_representation_size(self.n_sync_out)
        
        # DEBUG
        for synch_type, size in (('action', self.sync_representation_size_action), ('out', self.sync_representation_size_out)):
            print(f"Synch representation size {synch_type}: {size}")

        if self.sync_representation_size_action:  # if not zero
            self.set_synchronisation_parameters('action', self.n_sync_action, n_random_pairing_self)
        
        self.set_synchronisation_parameters('out', self.n_sync_out, n_random_pairing_self)

        # --- Output Projections ---
        self.d_output = d_output
        self.out_proj = nn.Sequential(nn.LazyLinear(self.d_output)) # No layer norm here? TODO: Check this
    
    def get_backbone(self, backbone: BackBone):
        # backbone._initialize(d_input=self.d_input)
        return backbone

    def get_synapses(self, synapses: Synapses):
        # synapses._initialize(d_model=self.d_model)
        return synapses

    def get_neuron_lvl_models(self, nlm: NLM):
        # nlm._initialize(d_memory=self.d_memory, d_model=self.d_model)
        return nlm

    def calculate_sync_representation_size(self, n_sync):
        """
        Calculate the size of the synchronisation representation based on neuron selection type.
        """
        if self.neuron_select_type == 'random-pairing':
            sync_representation_size = n_sync
        elif self.neuron_select_type in ('first-last', 'random'):
            sync_representation_size = (n_sync * (n_sync + 1)) // 2
        else:
            raise ValueError(f"Invalid neuron selection type: {self.neuron_select_type}")
        return sync_representation_size

    def set_synchronisation_parameters(self, sync_type: str, n_sync: int, n_random_pairing_self: int = 0):
        """
        1. Set the buffers for selecting neurons so that these indices are saved into the model state_dict.
        2. Set the parameters for learnable exponential decay when computing synchronisation between all 
            neurons.
        """
        assert sync_type in ('out', 'action'), f"Invalid synch_type: {sync_type}"
        left, right = self.initialize_left_right_neurons(sync_type, self.d_model, n_sync, n_random_pairing_self)
        sync_representation_size = self.sync_representation_size_action if sync_type == 'action' else self.sync_representation_size_out
        self.register_buffer(f'{sync_type}_neuron_indices_left', left)
        self.register_buffer(f'{sync_type}_neuron_indices_right', right)
        self.register_parameter(f'decay_params_{sync_type}', nn.Parameter(torch.zeros(sync_representation_size), requires_grad=True))

    def initialize_left_right_neurons(self, sync_type, d_model, n_sync, n_random_pairing_self=0):
        """
        Initialize the left and right neuron indices based on the neuron selection type.
        This complexity is owing to legacy experiments, but we retain that these types of
        neuron selections are interesting to experiment with.
        """
        if self.neuron_select_type=='first-last':
            if sync_type == 'out':
                neuron_indices_left = neuron_indices_right = torch.arange(0, n_sync)
            elif sync_type == 'action':
                neuron_indices_left = neuron_indices_right = torch.arange(d_model-n_sync, d_model)

        elif self.neuron_select_type=='random':
            neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_sync))
            neuron_indices_right = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_sync))

        elif self.neuron_select_type=='random-pairing':
            assert n_sync > n_random_pairing_self, f"Need at least {n_random_pairing_self} pairs for {self.neuron_select_type}"
            neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_sync))
            neuron_indices_right = torch.concatenate((
                neuron_indices_left[:n_random_pairing_self],
                torch.from_numpy(
                    np.random.choice(np.arange(d_model), size=n_sync-n_random_pairing_self)
                )
            ))

        device = self.get_parameter('initial_activated_state').device
        return neuron_indices_left.to(device), neuron_indices_right.to(device)

    def compute_synchronisation(self, activated_state: torch.Tensor, decay_alpha: Optional[torch.Tensor], decay_beta: Optional[torch.Tensor], r: torch.Tensor, sync_type: str):

        if sync_type == 'action': # Get action parameters
            n_sync = self.n_sync_action
            neuron_indices_left = self.get_buffer(f'{sync_type}_neuron_indices_left')
            neuron_indices_right = self.get_buffer(f'{sync_type}_neuron_indices_right')
        elif sync_type == 'out': # Get input parameters
            n_sync = self.n_sync_out
            neuron_indices_left = self.get_buffer(f'{sync_type}_neuron_indices_left')
            neuron_indices_right = self.get_buffer(f'{sync_type}_neuron_indices_right')
        
        if self.neuron_select_type in ('first-last', 'random'):
            selected_left = activated_state[:, neuron_indices_left]
            selected_right = activated_state[:, neuron_indices_right]
            
            # Compute outer product of selected neurons, NOTE: Matrix Multipliation !!
            outer = selected_left.unsqueeze(2) * selected_right.unsqueeze(1) # (B, n_sync*n_sync)
            # Resulting matrix is symmetric, so we only need the upper triangle
            i, j = torch.triu_indices(n_sync, n_sync) # (B, n_sync*(n_syn+1)/2)
            pairwise_product = outer[:, i, j]
            
        elif self.neuron_select_type == 'random-pairing':
            # For random-pairing, we compute the sync between specific pairs of neurons
            left = activated_state[:, neuron_indices_left]
            right = activated_state[:, neuron_indices_right]
            pairwise_product = left * right # NOTE: dot product!!
        else:
            raise ValueError("Invalid neuron selection type")
        
        # Compute synchronisation recurrently!! NOTE: JUST LIKE RNNs!!
        if decay_alpha is None or decay_beta is None:
            decay_alpha = pairwise_product
            decay_beta = torch.ones_like(pairwise_product)
        else:
            decay_alpha = r * decay_alpha + pairwise_product
            decay_beta = r * decay_beta + 1
        
        synchronisation = decay_alpha / (torch.sqrt(decay_beta)) 
        """NOTE:
        Decay Alpha are nothing but the Post-Activations calculated like RNNs fashion
        Decay Betas are for stability and so that decay_alpha don't shoot up eventually
        """
        return synchronisation, decay_alpha, decay_beta

    def compute_certainty(self, current_prediction: torch.Tensor):
        ne = compute_normalized_entropy(current_prediction)
        return torch.stack((ne, 1-ne), dim=-1)
    
    def compute_kv(self, x: torch.Tensor):
        # x.shape : (B, C, H, W) OR (B, emb, seq_len)
        input_features: torch.Tensor = self.backbone(x) # TODO: Make is more elegant the use of UNET -> DONE!!
        kv = self.kv_proj(
            input_features
                .flatten(start_dim=2) # (B, C, H*W) OR (B, emb, seq_len)
                .transpose(1, 2)      # (B, H*W, C) OR (B, seq_len, emb)
        )                             
        return kv # (B, H*W, d_input) OR (B, seq_len, d_input)

    def forward(self, x: torch.Tensor, track: bool = False):
        # x could be of dimension (B, n_ticks, d_emb, seq_len) OR (B, n_ticks, C, H, W)
        B = x.size(0)
        device = x.device

        # --- Initializing Tracking ---
        pre_activations_tracking = []
        post_activation_tracking = []
        attention_tracking = []
        sync_out_tracking = []          #TODO: find out how is this useful -> How one neuron activation impacts other neuron's acivations
        sync_action_tracking = []       #TODO: find out how is this useful
        
        # --- Initialize the temporal states z_0 & A_0---
        # NOTE We are using expand instead of repeat as done in loss_mnist fn because here the dimension to be expanded is 1 and since expand works on dim=1 so that it does'nt have to assign new memory, the memory is kept same as that of initial.
        activated_state = self.get_parameter('initial_activated_state').unsqueeze(0).expand((B, -1)) # (B, d_model)
        # pre_activation_history = self.get_parameter('initial_activation_history') # (B, d_model, d_mem)
        pre_activation_history = self.initial_activation_history.unsqueeze(0).expand((B, -1, -1)) # (B, d_model, d_mem) -> NOTE: used the self.'...' instead of above becuase of Tensor being assigned to a Parameter and LSP giving weird errors, techinically it's same!

        # --- Storage for Predictions & Certainty used later for loss ---
        predictions = torch.empty((B, self.d_output, self.n_ticks), dtype=x.dtype, device=device)
        certainties = torch.empty((B, 2, self.n_ticks), dtype=x.dtype, device=device)

        # --- Initialize Recurrent synchronization parameters ---
        decay_alpha_action, decay_beta_action = None, None # will be initialized automatically

        # initialize the learnable decay parameters and clamp the exp between [e^-0, e^-15]: NOTE: 15 is hyperparameter & NOTE Since we are clamping the data, we are not calling the ClampBackwards function during the update
        self.get_parameter('decay_params_action').data = self.get_parameter('decay_params_action').clamp(0, 15) 
        self.get_parameter('decay_params_out').data = self.get_parameter('decay_params_out').clamp(0, 15)

        # r_actions and r_out are the exp decay factors
        r_action, r_out = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(B, 1), torch.exp(-self.decay_params_out).unsqueeze(0).repeat(B, 1)
        
        # NOTE: We do this because the action synchronisation is one step ahead in time than that of output synchronization!!
        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, None, None, r_out, sync_type='out')
        
        # --- Temporal Loop ---
        for tick in range(self.n_ticks):
            # --- Getting the KV for DYNAMIC input data --- #NOTE: NOT CACHING IT RN
            kv = self.compute_kv(x[:, tick, ...]) # Running the KV for current tick's input

            # --- Calculate the Synchronization Action --- 
            sync_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(activated_state, decay_alpha_action, decay_beta_action, r_action, sync_type='action')
            # sync_action (S_action): (B, sync_action_repr_size)
            # --- Apply Cross-Attention of Synchronized Action & input data ---
            q = self.q_proj(sync_action).unsqueeze(1) # (B, 1, d_input) -> projecting the flattened synchronized action to d_input shape
            attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True) # Returns the attention weights perhead w/o averaging, shapes are (B, C=1, d_input), (B, n_head, C=1, H*W/seq_len)
            attn_out = attn_out.squeeze(1) # (B, 1, d_input) -> (B, d_input)
            pre_synapse_input = torch.cat((attn_out, activated_state), dim=-1) # (B, d_input+d_model)

            # --- Apply Synapse ---
            pre_activation = self.synapses(pre_synapse_input) # (B, d_model)
            pre_activation_history = torch.cat((pre_activation_history[:, :, 1:], pre_activation.unsqueeze(-1)), dim=-1)

            # --- Apply NLMs / History Processor ---
            activated_state = self.history_processor(pre_activation_history) # (B, d_model)

            # --- Calculate Synchronization Output ---
            sync_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, decay_alpha_out, decay_beta_out, r_out, sync_type='out')
            # sync_out (S_out): (B, sync_out_repr_size)

            # -- Get the Predictions ---
            current_predition = self.out_proj(sync_out) # (B, d_out)
            current_certainty = self.compute_certainty(current_predition) # (B, 2) -> Includes both normalized and 1 - normalized entropy

            predictions[:, :, tick] = current_predition
            certainties[:, :, tick] = current_certainty

            # --- Tracking ---
            if track:
                pre_activations_tracking.append(pre_activation_history[:, :, -1].detach().cpu().numpy())
                post_activation_tracking.append(activated_state.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())
                sync_action_tracking.append(sync_action.detach().cpu().numpy())
                sync_out_tracking.append(sync_out.detach().cpu().numpy())
        
        # --- Return Values ---
        if track:
            return predictions, certainties, (
                np.array(pre_activations_tracking),
                np.array(post_activation_tracking),
                np.array(attention_tracking),
                np.array(sync_action_tracking),
                np.array(sync_out_tracking)
            )

        return predictions, certainties, (
            self.get_parameter('decay_params_action').data,
            self.get_parameter('decay_params_out').data
        )
        


 
