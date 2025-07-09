import torch
import torch.nn.functional as F


def compute_normalized_entropy(logits: torch.Tensor, reduction='mean'):
    """Compute the normalized entropy for certainity-loss."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = - torch.sum(probs * log_probs, dim=-1)
    num_classes = logits.shape[-1]
    max_entropy =  torch.log(torch.tensor(num_classes, dtype=logits.dtype, device=logits.device))
    normalized_entropy = entropy /  max_entropy

    if len(logits.shape) > 2 and reduction == 'mean': # when different classes across different channels for example 
        normalized_entropy = normalized_entropy.flatten(start_dim=1).mean(-1)
    return normalized_entropy

def calculate_accuracy(predictions_logits: torch.Tensor, targets: torch.Tensor, most_certain_tick: torch.Tensor):
    """ Calculate the Accuracy using the prediction at most certain tick"""
    device = predictions_logits.device
    B = predictions_logits.shape[0]
    # TODO: Check why running following is giving CUDA OOM and not the one below that!!
    # preds_at_tick_class_idx = predictions_logits[:, :, most_certain_tick].argmax(dim=1).detach().cpu().numpy() # Returns the idx for max prob class of shape (B)
    preds_at_tick_class_idx = predictions_logits.argmax(1)[torch.arange(B, device=device), most_certain_tick].detach().cpu().numpy()
    accuracy = (targets.detach().cpu().numpy() == preds_at_tick_class_idx).mean()

    return accuracy
    
def visualize_attention_map_with_images():
    # STEP1: Get the output of an image from the Model, along with the attention maps for all heads per ticks along with certainities & predictions
    # STEP2: Unroll the attention weights and rehape it from H*W to (H, W).
    # STEP3: Perstep get the average pixel of attention across all the weights and mark it as a point
    # STEP4: display all these traces and attention maps along with image for all the ticks along with the preds and certainities
    pass
