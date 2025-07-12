import torch
import imageio
import numpy as np
import torch.nn.functional as F
import cv2
import imageio
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.ctm_dyn_kv import ContinousThoughtMachineDyn

def unnormalize(name:str, image_array: np.ndarray):
    if name == 'MNIST':
        mean = np.array([0.0]).reshape(1, 1, 1)
        std = np.array([1.0]).reshape(1, 1, 1)
        img = (image_array* std + mean).transpose(1, 2, 0)
    elif name == 'CIFAR10':
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
        std = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)
        img = (image_array* std + mean).transpose(1, 2, 0)
    elif name == 'MNIST_COUNT':
        mean = np.array([0.0]).reshape(1, 1, 1, 1) # [ticks, channels, H, W]
        std = np.array([1.0]).reshape(1, 1, 1, 1)
        img = (image_array* std + mean).transpose(0, 2, 3, 1)
    elif name == 'CIFAR10_COUNT':
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1) # [ticks, channels, H, W]
        std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1)
        img = (image_array* std + mean).transpose(0, 2, 3, 1)
    else:
        raise ValueError(f"Unnormalized is not defined for {name}")
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def visualize_attention_map_with_images(name, model, input, label, label_text, device, path):
    
    if device == "cuda":
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    import os
    if not os.path.exists(path):
        os.mkdir(path)

    total_len = input.shape[0]
    assert label.shape[0] == total_len, "Number of Labels must be equal to number of inputs"
    
    model = model.to(device)
    model.eval()
    if isinstance(model, ContinousThoughtMachineDyn):
        predictions, certainties, (
            pre_activations_tracking,
            post_activation_tracking,
            attention_tracking,             # [n_ticks, B, n_heads, n_channel, H*W]
            sync_action_tracking,
            sync_out_tracking
        ) = model(input.to(device), track=True, static_input=True)
    else:
        predictions, certainties, (
            pre_activations_tracking,
            post_activation_tracking,
            attention_tracking,             # [n_ticks, B, n_heads, n_channel, H*W]
            sync_action_tracking,
            sync_out_tracking
        ) = model(input.to(device), track=True)

    predictions = predictions.detach()
    certainties = certainties.detach()

    n_ticks, B, n_heads, C, seq_len = attention_tracking.shape
    print("Attention Shape: ", attention_tracking.shape)
    H = int(seq_len ** 0.5)
    while seq_len % H != 0: H -= 1
    W = seq_len // H # To make compensate for non-square seq length
    attention_tracking = np.reshape(attention_tracking, [n_ticks, B, n_heads, C, H, W])
    #TODO: Interpolate the attention if not square ?? but why??
        
    for idx in range(total_len):
        create_gif(
            name=name,
            image=input[idx],
            predictions=predictions[idx],
            certainties=certainties[idx],
            attention_tracks=attention_tracking[:, idx, :, :, :],
            label=label[idx],
            label_text=label_text,
            gif_path=f'{path}/output_{idx}.gif',
            fps=int(attention_tracking.shape[0]/(2.8)),
        )

def visualize_attention_map_with_dynamic_images(name, model, input, label, label_text, device, path):
    
    if device == "cuda":
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    import os
    if not os.path.exists(path):
        os.mkdir(path)

    total_len = input.shape[0]
    assert label.shape[0] == total_len, "Number of Labels must be equal to number of inputs"

    model = model.to(device)
    model.eval()
    predictions, certainties, (
        pre_activations_tracking,
        post_activation_tracking,
        attention_tracking,             # [n_ticks, B, n_heads, n_channel, H*W]
        sync_action_tracking,
        sync_out_tracking
    ) = model(input.to(device), track=True)
    predictions = predictions.detach()
    certainties = certainties.detach()
    
    n_ticks, B, n_heads, C, seq_len = attention_tracking.shape
    print("Attention Shape: ", attention_tracking.shape)
    H = int(seq_len ** 0.5)
    while seq_len % H != 0: H -= 1
    W = seq_len // H # To make compensate for non-square seq length
    attention_tracking = np.reshape(attention_tracking, [n_ticks, B, n_heads, C, H, W])
    #TODO: Interpolate the attention if not square ?? but why??
        
    for idx in range(total_len):
        create_gif(
            name=name,
            image=input[idx],
            predictions=predictions[idx],
            certainties=certainties[idx],
            attention_tracks=attention_tracking[:, idx, :, :, :],
            label=label[idx],
            label_text=label_text,
            gif_path=f'{path}/output_{idx}.gif',
            fps=int(attention_tracking.shape[0]/(20)),
            dyn=True
        )


def create_gif(
    name,
    image,
    predictions,
    certainties,
    attention_tracks,
    label,
    label_text,
    gif_path,
    fps=10,
    max_heads=4,
    dyn=False
):
    num_classes, n_ticks = predictions.shape
    n_heads = attention_tracks.shape[1]
    H, W = attention_tracks.shape[-2:]

    # Unnormalize and prepare image
    img_array = image.cpu().numpy()
    img = unnormalize(name=name, image_array=img_array)
    
    # Keep original image size for better quality
    RESIZED_X = 28
    RESIZED_Y = 28
    
    img_resized = None

    pred = F.softmax(predictions, dim=0).cpu().numpy()  # [num_classes, n_ticks]
    cert = certainties[1].cpu().numpy()  # [n_ticks]

    # path_coords = []
    frames = []

    for t in range(n_ticks):
        img_tick = img[t] if dyn else img
        correct_class = int(label[t]) if dyn else int(label)

        if img_resized is None or dyn:
            img_resized = cv2.resize(img_tick, (RESIZED_X, RESIZED_Y))

        # Create figure with tight layout and no padding
        fig = plt.figure(figsize=(12, 6), dpi=100, facecolor='white')
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.15, hspace=0.15)
        
        gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1], height_ratios=[1, 1])

        # Step 1: Input Image (top-left)
        ax_img = fig.add_subplot(gs[0, 0])
        base_img = img_resized.copy()

        attn = attention_tracks[t]  # [n_heads, 1, H, W]
        attn_heads = attn.squeeze(1)   # [n_heads, H, W]
        # Normalize the attention
        attn_heads_min = attn_heads.min(axis=(1, 2), keepdims=True)
        attn_heads_max = attn_heads.max(axis=(1, 2), keepdims=True)
        # print(attn_heads_max.shape)
        attn_heads = (attn_heads - attn_heads_min)/(attn_heads_max - attn_heads_min)
        # avg_attn = np.mean(attn_heads, axis=0) # [H, W]

        # print(avg_attn)
        # print(np.argmax(avg_attn))
        # yx = np.unravel_index(np.argmax(avg_attn), avg_attn.shape)
        # max_x, max_y = int(yx[1] * RESIZED_X / W), int(yx[0] * RESIZED_Y / H)
        # path_coords.append((max_x, max_y))

        # Draw attention path lines on the image
        if len(base_img.shape) == 2:
            ax_img.imshow(base_img, cmap='gray')
        else:
            ax_img.imshow(base_img)
        # if len(path_coords) > 1:
        #     xs, ys = zip(*path_coords)
        #     ax_img.plot(xs, ys, color='cyan', linewidth=2)
        # # Draw current attention max point
        # ax_img.scatter([max_x], [max_y], color='blue', s=50)
        #
        # ax_img.axis('off')
        ax_img.set_title('Input Image', fontsize=12, pad=10)

        # Step 2: Attention heatmaps (top-middle and top-right)
        heads_to_show = min(n_heads, max_heads)
        for i in range(heads_to_show):
            row = i // 4
            col = (i % 4) + 1  # Start from column 1 (after image)
            # if col < 4:  # Only show first 4 heads in a 2x2 grid
            ax_attn = fig.add_subplot(gs[row, col])
            head_attn = attn_heads[i]
            # Resize attention to match input image size for consistent aspect ratio
            head_resized = cv2.resize(head_attn, (RESIZED_X, RESIZED_Y))
            norm = cv2.normalize(head_resized, None, 0, 1, cv2.NORM_MINMAX)
            
            ax_attn.imshow(norm, cmap='viridis')
            ax_attn.axis('off')
            ax_attn.set_title(f'Head {i+1}', fontsize=10)

        # If we have space, add a combined attention view
        if heads_to_show <= 3:
            ax_combined = fig.add_subplot(gs[0, 4])
            combined_attn = np.mean(attn_heads[:heads_to_show], axis=0)
            combined_resized = cv2.resize(combined_attn, (RESIZED_X, RESIZED_Y))
            combined_norm = cv2.normalize(combined_resized, None, 0, 1, cv2.NORM_MINMAX)

            ax_combined.imshow(combined_norm, cmap='jet')
            ax_combined.axis('off')
            ax_combined.set_title('Average Attention', fontsize=10)

        # Step 3: Prediction bar chart (bottom-left and bottom-center)
        ax_pred = fig.add_subplot(gs[1, 0:3])
        bars = ax_pred.bar(range(num_classes), pred[:, t], color='lightgray')
        bars[correct_class].set_color('green')
        pred_class = np.argmax(pred[:, t])
        if pred_class != correct_class:
            bars[pred_class].set_color('red')
        
        ax_pred.set_ylim(0, 1)
        ax_pred.set_ylabel('Probability', fontsize=10)
        ax_pred.set_title(f'Predictions (Tick {t})', fontsize=12)
        ax_pred.set_xticks(range(num_classes))
        ax_pred.set_xticklabels(label_text, rotation=0, fontsize=8, ha='center')
        ax_pred.grid(True, alpha=0.3)

        # Step 4: Certainty line chart (bottom-right)
        ax_cert = fig.add_subplot(gs[1, 3:5])
        ax_cert.set_xlim(0, n_ticks-1)
        ax_cert.set_ylim(0, 1)
        ax_cert.set_title('Certainty over Time', fontsize=12)
        ax_cert.set_xlabel('Tick', fontsize=10)
        ax_cert.set_ylabel('Certainty', fontsize=10)
        ax_cert.grid(True, linestyle='--', alpha=0.3)

        ax_cert.plot(range(t+1), cert[:t+1], color='orange', linewidth=2)
        # Add current tick marker
        if t < len(cert):
            ax_cert.scatter([t], [cert[t]], color='red', s=50, zorder=5)

        # Convert to numpy array for GIF frame
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        canvas = fig.canvas
        if not isinstance(canvas, FigureCanvasAgg):
            canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.tostring_argb()
        w, h = canvas.get_width_height()
        img_array = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        img_array = img_array[:, :, [1, 2, 3, 0]]  # ARGB -> RGBA
        frame = img_array[:, :, :3]  # Drop alpha for RGB

        plt.close(fig)
        frames.append(frame)

    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"✅ GIF saved to {gif_path}")
