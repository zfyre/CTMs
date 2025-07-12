import torch
import torch.nn.functional as F
from typing import Optional
from collections import defaultdict
from torchvision import datasets, transforms

def prepare_data(name:str , batch_size:int, n_ticks:Optional[int], path:str = "./data"):
    
    if name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = datasets.MNIST(root=path, train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root=path, train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, num_workers=1)
        testloader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True, num_workers=1, drop_last=False)
        return trainloader, testloader
    elif name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])
        train_data = datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root=path, train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, num_workers=1)
        testloader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True, num_workers=1, drop_last=False)
        return trainloader, testloader
    elif name == 'MNIST_COUNT':
        if n_ticks is None:
            raise ValueError("n_ticks is required")
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = datasets.MNIST(root=path, train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root=path, train=False, download=True, transform=transform)
        
        def collate_running_mode_labels(batch):
            images, labels = zip(*batch)
            images = torch.stack(images)  # [B * n_ticks, 1, 28, 28]
            labels = torch.tensor(labels) # [B * n_ticks]
            
            total = len(images)
            valid_len = (total // n_ticks) * n_ticks  # truncate to multiple of n_ticks

            if valid_len == 0:
                return None  # batch too small to form even one sample

            images = images[:valid_len]
            labels = labels[:valid_len]

            B = valid_len // n_ticks
            images = images.view(B, n_ticks, 1, 28, 28)
            labels = labels.view(B, n_ticks)

            # Compute running mode with tie-breaking (larger number wins)
            new_labels = []
            for row in labels:
                counts = defaultdict(int)
                running_modes = []
                for i in range(len(row)):
                    counts[row[i].item()] += 1
                    max_freq = max(counts.values())
                    # find all items with max freq
                    candidates = [k for k, v in counts.items() if v == max_freq]
                    running_modes.append(max(candidates))  # break tie by max
                new_labels.append(running_modes)

            new_labels = torch.tensor(new_labels)  # [B, n_ticks]

            return images, new_labels

        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size*n_ticks, shuffle=True, num_workers=1, collate_fn=collate_running_mode_labels)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size*n_ticks, shuffle=True, num_workers=1, drop_last=False, collate_fn=collate_running_mode_labels)
        return trainloader, testloader
    elif name == 'CIFAR10_COUNT':
        if n_ticks is None:
            raise ValueError("n_ticks is required")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            ),
            transforms.Resize([28, 28])
        ])
        train_data = datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root=path, train=False, download=True, transform=transform)
        
        def collate_running_mode_labels(batch):
            images, labels = zip(*batch)
            images = torch.stack(images)  # [B * n_ticks, 1, 28, 28]
            labels = torch.tensor(labels) # [B * n_ticks]
            
            total = len(images)
            valid_len = (total // n_ticks) * n_ticks  # truncate to multiple of n_ticks

            if valid_len == 0:
                return None  # batch too small to form even one sample

            images = images[:valid_len]
            labels = labels[:valid_len]

            B = valid_len // n_ticks
            images = images.view(B, n_ticks, 3, 28, 28)
            labels = labels.view(B, n_ticks)

            # Compute running mode with tie-breaking (larger number wins)
            new_labels = []
            for row in labels:
                counts = defaultdict(int)
                running_modes = []
                for i in range(len(row)):
                    counts[row[i].item()] += 1
                    max_freq = max(counts.values())
                    # find all items with max freq
                    candidates = [k for k, v in counts.items() if v == max_freq]
                    running_modes.append(max(candidates))  # break tie by max
                new_labels.append(running_modes)

            new_labels = torch.tensor(new_labels)  # [B, n_ticks]

            return images, new_labels

        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size*n_ticks, shuffle=True, num_workers=1, collate_fn=collate_running_mode_labels)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size*n_ticks, shuffle=True, num_workers=1, drop_last=False, collate_fn=collate_running_mode_labels)
        return trainloader, testloader
    else:
        raise ValueError(f"Dataset {name} not found")

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

def calculate_accuracy_mnist_count(predictions_logits: torch.Tensor, targets: torch.Tensor, most_certain_tick: torch.Tensor):
    """ Calculate the Accuracy using the prediction at most certain tick"""
    device = predictions_logits.device
    B = predictions_logits.shape[0]
    # TODO: Check why running following is giving CUDA OOM and not the one below that!!
    # preds_at_tick_class_idx = predictions_logits[:, :, most_certain_tick].argmax(dim=1).detach().cpu().numpy() # Returns the idx for max prob class of shape (B)
    preds_at_tick_class_idx = predictions_logits.argmax(1)[torch.arange(B, device=device), most_certain_tick].detach().cpu().numpy()
    targets_at_tick_class_idx = targets[torch.arange(B, device=device), most_certain_tick].detach().cpu().numpy()
    accuracy = (targets_at_tick_class_idx == preds_at_tick_class_idx).mean()

    return accuracy


