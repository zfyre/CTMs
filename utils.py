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