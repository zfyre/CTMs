import torch
import torch.nn as nn

def loss_mnist_(predictions_logits: torch.Tensor, certainities: torch.Tensor, targets: torch.Tensor):
    # NOTE Cross Entorpy in torch internally applied logSoftmax, hence no need for applyting Softmax on logits
    criterion = nn.CrossEntropyLoss(reduction='none')
    losses: torch.FloatTensor = criterion(
        predictions_logits,
        targets
            .unsqueeze(-1)
            .repeat((1, predictions_logits.shape[-1])),
    ) # Returns losses of shape (B, n_ticks)

    loss_idx_1 = losses.argmin(dim=1) # Returns indices for t1 of shape (B)
    loss_idx_2 = certainities[:, 1].argmax(dim=-1) # Returns indices for t2 of shape (B)
    
    loss_t1 = losses[:, loss_idx_1].mean()
    loss_t2 = losses[:, loss_idx_2].mean()

    # TODO: Check if this is any different from the above & is not the reason for no accuracy
    # batch_indexer = torch.arange(predictions_logits.size(0), device=predictions_logits.device)
    # loss_t1 = losses[batch_indexer, loss_idx_1].mean()
    # loss_t2 = losses[batch_indexer, loss_idx_2].mean()

    loss = (loss_t1 + loss_t2)/2
        
    return loss, loss_idx_2


    
