import io
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


def prepare_data(name, batch_size, path = "./data"):
    
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
    
def visualize_attention_map_with_images(model, input, label, label_text, device):
    # STEP3: Perstep get the average pixel of attention across all the weights and mark it as a point
    # STEP4: display all these traces and attention maps along with image for all the ticks along with the preds and certainities
    
    # STEP1: Get the output of an image from the Model, along with the attention maps for all heads per ticks along with certainities & predictions
    model = model.to(device)
    model.eval()
    predictions, certainties, (
        pre_activations_tracking,
        post_activation_tracking,
        attention_tracking,
        sync_action_tracking,
        sync_out_tracking
    ) = model(input.to(device), track=True)
    
    # Visualize the Certainities & Prediction with Ticks
    predictions = predictions.detach()
    certainties = certainties.detach()
    create_prediction_certainty_gif(input, predictions, certainties, label, label_text, 'output.gif')

    # STEP2: Unroll the attention weights and reshape it from H*W to (H, W).
    

def create_prediction_certainty_gif(image, predictions, certainties, label, label_text, gif_path, fps=5):
    """
    Creates a GIF showing input image, class predictions (bar chart), and certainty (line graph) over time.
    
    Args:
        image: Tensor of shape [B, 3, H, W], assumed to be normalized as in CIFAR10.
        predictions: Tensor of shape [B, num_classes, n_ticks]
        certainties: Tensor of shape [B, 2, n_ticks]
        label: Tensor of shape [B, 1] or [B]
        gif_path: Path to save the output GIF
        fps: Frames per second for the GIF
    """
    # === Unpack shapes ===
    B, num_classes, n_ticks = predictions.shape
    assert B == 1, "Only supports batch size of 1"
    
    # === Unnormalize image ===
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    img_tensor = image[0].cpu().clone()
    for t, m, s in zip(img_tensor, mean, std):
        t.mul_(s).add_(m)
    img = img_tensor.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)  # Ensure values in [0, 1] for display

    # === Extract prediction and certainty info ===
    pred = F.softmax(predictions[0], dim=0).cpu().numpy()   # [num_classes, n_ticks]
    cert = certainties[0, 1].cpu().numpy()                  # [n_ticks]
    correct_class = int(label[0])                           # scalar

    frames = []

    for t in range(n_ticks):
        fig = plt.figure(figsize=(9, 6))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=[2, 1])

        # === Show image ===
        ax_img = fig.add_subplot(gs[:, 0])
        ax_img.imshow(img)
        ax_img.axis('off')
        ax_img.set_title('Input Image')

        # === Bar plot: predictions ===
        ax1 = fig.add_subplot(gs[0, 1])
        bar_colors = ['skyblue'] * num_classes
        predicted_class = int(np.argmax(pred[:, t]))
        
        if predicted_class != correct_class:
            bar_colors[predicted_class] = 'red'
        bar_colors[correct_class] = 'green'  # Override to green if same
        
        ax1.bar(range(num_classes), pred[:, t], color=bar_colors)
        ax1.set_ylim(0, 1)
        ax1.set_title(f'Log-Scaled Predictions at tick {t} (Label: {label_text[correct_class]})')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Prob')
        ax1.set_xticks(range(num_classes))
        ax1.set_xticklabels(label_text, rotation=45, ha='right', fontsize=8)

        # === Line plot: certainty ===
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.plot(range(t + 1), cert[:t + 1], color='orange')
        ax2.set_xlim(0, n_ticks - 1)
        ax2.set_ylim(0, 1)
        ax2.set_title('Certainty over time')
        ax2.set_xlabel('Tick')
        ax2.set_ylabel('Certainty')

        fig.tight_layout()

        # Save frame to memory
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()
        plt.close(fig)

    # === Save all frames into a gif ===
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"✅ GIF saved to {gif_path}")

