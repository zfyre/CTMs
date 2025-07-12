import torch
import torch.nn as nn

def loss_classifier_(predictions_logits: torch.Tensor, certainities: torch.Tensor, targets: torch.Tensor):
    # NOTE Cross Entorpy in torch internally applied logSoftmax, hence no need for applyting Softmax on logits
    criterion = nn.CrossEntropyLoss(reduction='none')
    losses: torch.FloatTensor = criterion(
        predictions_logits, # [B, d_output, n_ticks]
        targets # [B]
            .unsqueeze(-1)
            .repeat((1, predictions_logits.shape[-1])), # [B, n_ticks]
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
    # loss = losses.mean() #NOTE: For testig what happens when we do this!
    # loss = (losses.mean() + loss_t2)/2 #NOTE: For testing what happens when we do this!
        
    return loss, (loss_idx_1, loss_idx_2)

def loss_mnist_count_(predictions_logits: torch.Tensor, certainities, targets: torch.Tensor):
    # NOTE Cross Entorpy in torch internally applied logSoftmax, hence no need for applyting Softmax on logits
    criterion = nn.CrossEntropyLoss(reduction='none')
    losses: torch.FloatTensor = criterion(
        predictions_logits, # [B, d_output, n_ticks]
        targets # [B, n_ticks]
    ) # Returns losses of shape (B, n_ticks)

    loss_idx_1 = losses.argmin(dim=1) # Returns indices for t1 of shape (B)
    loss_idx_2 = certainities[:, 1].argmax(dim=-1) # Returns indices for t2 of shape (B)
    
    loss_t1 = losses[:, loss_idx_1].mean()
    loss_t2 = losses[:, loss_idx_2].mean()
    
    #TODO: 1) Certainity weighted LOSS 2) Only taking final tick loss 3) combined ticks loss 4) The above classifier loss (legacy loss)

    loss = (loss_t1 + loss_t2)/2 #NOTE: (4)
    # loss = losses.mean() #NOTE: (3)
    # loss = losses[:, -1].mean() #NOTE: (2)
    # loss = (certainities[:, 1] * losses).mean() #NOTE: (1)
        
    return loss, (loss_idx_1, loss_idx_2)


