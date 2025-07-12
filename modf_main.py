import torch
import argparse
import json
import os
import numpy as np
from datetime import datetime
# import matplotlib
# matplotlib.use('Agg')  # Use Anti-Grain Geometry backend (no GUI)
import matplotlib.pyplot as plt
from models.ctm import ContinousThoughtMachine
from models.mnist import BackBone, NLM, Synapses
from utils import prepare_data
from train import train_classifier

def create_run_directory(base_path="./data/runs"):
    """Create a unique timestamped directory for this run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_path, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def get_model_info(model):
    """Get model information, handling uninitialized parameters."""
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_mb,
            "initialization_status": "initialized"
        }
    except ValueError as e:
        if "uninitialized parameter" in str(e):
            return {
                "total_parameters": "uninitialized",
                "trainable_parameters": "uninitialized", 
                "model_size_mb": "uninitialized",
                "initialization_status": "uninitialized",
                "note": "Model contains lazy modules that need initialization"
            }
        else:
            raise e

def save_model_config(run_dir, model, args):
    """Save model configuration and training arguments."""
    config = {
        "backbone_config": {
            "d_input": args.numd_input,
            "use_unet": args.use_unet
        },
        "synapses_config": {
            "d_model": args.numd_model
        },
        "nlm_config": {
            "d_memory": args.num_memory,
            "d_model": args.numd_model,
            "memory_hidden_dims": args.numd_hidden_mem
        },
        "model_config": {
            "n_ticks": args.num_ticks,
            "d_model": args.numd_model,
            "d_memory": args.num_memory,
            "n_sync_out": args.num_sync_out,
            "n_sync_action": args.num_sync_action,
            "d_input": args.numd_input,
            "n_heads": args.num_heads,
            "d_output": args.numd_output,
            "dropout": args.dropout,
            "neuron_selection_type": args.selection_type,
            "n_random_pairing_self": args.num_rand_self_pair,
        },
        "training_args": {
            "name": args.name,
            "batch_size": args.batch_size,
            "path": args.path,
            "iterations": args.iterations,
            "lr": args.lr,
            "device": args.device,
            "seed": args.seed,
            "num_ticks": args.num_ticks,
            "num_memory": args.num_memory,
        },
        "model_info": get_model_info(model),
        "timestamp": datetime.now().isoformat(),
    }
    
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path

def convert_to_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to JSON-serializable format."""
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj

def save_training_results(run_dir, model, train_losses_list, least_loss_tick_list, most_certain_tick_list):
    """Save model weights and training metrics."""
    
    # Save Backbone state dict
    model_backbone_path = os.path.join(run_dir, "model_backbone_weights.pth")
    torch.save(model.backbone.state_dict(), model_backbone_path)
    
    # Save model state dict
    model_path = os.path.join(run_dir, "model_weights.pth")
    torch.save(model.state_dict(), model_path)

    # Save Backbone state dict
    model_full_backbone_path = os.path.join(run_dir, "model_full_backbone.pth")
    torch.save(model.backbone, model_full_backbone_path)
    
    # Save complete model (architecture + weights) for easier loading
    full_model_path = os.path.join(run_dir, "full_model.pth")
    torch.save(model, full_model_path)
    
    # Convert all data to JSON-serializable format
    metrics = {
        # "train_losses": convert_to_serializable(train_losses_list),
        # "least_loss_tick_list": convert_to_serializable(least_loss_tick_list), TODO: Not saving these rightnow because they will take huge memory unncessarily
        # "most_certain_tick_list": convert_to_serializable(most_certain_tick_list),
        "final_loss": convert_to_serializable(train_losses_list[-1]) if train_losses_list else None,
        "best_loss": convert_to_serializable(min(train_losses_list)) if train_losses_list else None,
        "best_loss_iteration": train_losses_list.index(min(train_losses_list)) if train_losses_list else None,
    }
    
    metrics_path = os.path.join(run_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save final model info (now that parameters are initialized)
    final_model_info = get_model_info(model)
    final_model_info_path = os.path.join(run_dir, "final_model_info.json")
    with open(final_model_info_path, 'w') as f:
        json.dump(final_model_info, f, indent=2)
    
    return model_path, full_model_path, metrics_path, final_model_info_path, model_full_backbone_path

def create_plots(run_dir, train_losses_list, least_loss_tick_list, most_certain_tick_list):
    """Create and save training plots."""
    plots_created = []
    
    # Training loss plot
    if train_losses_list and len(train_losses_list) > 0:
        plt.figure(figsize=(12, 8))
        
        # Main loss plot
        plt.subplot(2, 2, 1)
        plt.plot(train_losses_list, 'b-', linewidth=1.5)
        plt.title("Training Loss Over Time")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        
        # Log scale loss plot
        plt.subplot(2, 2, 2)
        plt.semilogy(train_losses_list, 'r-', linewidth=1.5)
        plt.title("Training Loss (Log Scale)")
        plt.xlabel("Iteration")
        plt.ylabel("Loss (log scale)")
        plt.grid(True, alpha=0.3)
        
        # Least loss tick analysis
        if least_loss_tick_list and len(least_loss_tick_list) > 0:
            plt.subplot(2, 2, 3)
            # Create x-axis values for each batch across all iterations
            x_vals = []
            y_vals = []
            avg_ticks = []
            
            for iteration, ticks in enumerate(least_loss_tick_list):
                if isinstance(ticks, (list, np.ndarray)):
                    # Multiple ticks per iteration (one per batch)
                    x_vals.extend([iteration] * len(ticks))
                    y_vals.extend(ticks)
                    avg_ticks.append(np.mean(ticks))
                else:
                    # Single tick per iteration
                    x_vals.append(iteration)
                    y_vals.append(ticks)
                    avg_ticks.append(ticks)
            
            plt.scatter(x_vals, y_vals, c='lightgreen', alpha=0.6, s=10, label='Individual Batches')
            plt.plot(range(len(avg_ticks)), avg_ticks, 'g-', linewidth=2, label='Average')
            plt.title("Least Loss Tick Over Time")
            plt.xlabel("Iteration")
            plt.ylabel("Tick Number")
            plt.grid(True, alpha=0.3)
            plt.legend()

        # Most certain tick analysis
        if most_certain_tick_list and len(most_certain_tick_list) > 0:
            plt.subplot(2, 2, 4)
            # Create x-axis values for each batch across all iterations
            x_vals = []
            y_vals = []
            avg_ticks = []
            
            for iteration, ticks in enumerate(most_certain_tick_list):
                if isinstance(ticks, (list, np.ndarray)):
                    # Multiple ticks per iteration (one per batch)
                    x_vals.extend([iteration] * len(ticks))
                    y_vals.extend(ticks)
                    avg_ticks.append(np.mean(ticks))
                else:
                    # Single tick per iteration
                    x_vals.append(iteration)
                    y_vals.append(ticks)
                    avg_ticks.append(ticks)
            
            plt.scatter(x_vals, y_vals, c='plum', alpha=0.6, s=10, label='Individual Batches')
            plt.plot(range(len(avg_ticks)), avg_ticks, 'm-', linewidth=2, label='Average')
            plt.title("Most Certain Tick Over Time")
            plt.xlabel("Iteration")
            plt.ylabel("Tick Number")
            plt.grid(True, alpha=0.3)
            plt.legend()       

        plt.tight_layout()
        loss_plot_path = os.path.join(run_dir, "training_analysis.png")
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(loss_plot_path)
        
        # Create a simple loss plot as well
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses_list, 'b-', linewidth=1.5)
        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        simple_loss_path = os.path.join(run_dir, "simple_loss_plot.png")
        plt.savefig(simple_loss_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(simple_loss_path)
    
    return plots_created

def save_summary(backbone_path, desc, run_dir, config_path, model_path, full_model_path, metrics_path, final_model_info_path, plots_created):
    """Save a summary of all saved files."""
    summary = {
        "description": desc,
        "run_directory": run_dir,
        "files_saved": {
            "config": config_path,
            "model_weights": model_path,
            "full_model": full_model_path,
            "training_metrics": metrics_path,
            "model_full_backbone": backbone_path,
            "final_model_info": final_model_info_path,
            "plots": plots_created,
        },
        "saved_at": datetime.now().isoformat(),
    }
    
    summary_path = os.path.join(run_dir, "run_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--message", type=str, required=True, help="short description of the run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # Training Params
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--name", type=str, default="MNIST")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--path", type=str, default="./data")
    
    # Backbone Parameters
    parser.add_argument("--backbone", type=str, help="The path of the Backbone model")
    parser.add_argument("--backbone_config", type=str, help="The path of the Backbone args config")
    parser.add_argument("--use_unet", type=bool, default=True)
    
    # Syanpses Parameters

    # Nlm Parameters
    parser.add_argument("--numd_hidden_mem", type=int, default=8)
    
    # CTM Parameters
    parser.add_argument("--num_ticks", type=int, default=32)
    parser.add_argument("--numd_model", type=int, default=300)
    parser.add_argument("--num_memory", type=int, default=15)
    parser.add_argument("--num_sync_out", type=int, default=80)
    parser.add_argument("--num_sync_action", type=int, default=80)
    parser.add_argument("--numd_input", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--numd_output", type=int, default=10)
    parser.add_argument("--dropout", type=int, default=0)
    parser.add_argument("--selection_type", type=str, default="first-last")
    parser.add_argument("--num_rand_self_pair", type=int, default=0)
    
    args = parser.parse_args()

    device = torch.device(args.device)
    
    # --- Set Seed ---
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    # --- Create Run Directory ---
    run_dir = create_run_directory()
    print(f"Created run directory: {run_dir}")
    
    # --- Prepare Data ---
    print(f"Preparing {args.name} Dataset")
    trainloader, testloader = prepare_data(args.name, args.batch_size, args.path)
    
    # --- Prepare Model ---
    assert(args.num_ticks >= args.num_memory)
    
    # backbone, synapses, nlm model's configuration will be modified accordingly inside CTM hence initialized with 'zeros'
    if args.backbone:
        print(f"Loading backbone from path={args.backbone}")
        backbone_data = torch.load(args.backbone, map_location=device)
        if isinstance(backbone_data, BackBone):
            backbone = backbone_data
        else:
            if not args.backbone_config:
                raise ValueError("Need the Backbone Config to load model from state_dict()")
            with open(args.backbone_config, 'r') as f:
                backbone_config = json.load(f)
            backbone = BackBone(
                d_input=backbone_config['backbone_config']['d_input'],
                use_unet=backbone_config['backbone_config']['use_unet']
            )
            backbone.load_state_dict(backbone_data)

        backbone = backbone.to(device)
        # Freeze all layers
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()
        print(backbone)
    else:
        print(f"Initializing a new backbone with UNET={args.use_unet}")
        backbone = BackBone(
            d_input=args.numd_input,
            use_unet=args.use_unet
        ).to(device)
    
    synapses = Synapses(
        d_model=args.numd_model
    ).to(device)
    nlm = NLM(
        d_memory=args.num_memory,
        d_model=args.numd_model,
        memory_hidden_dims=args.numd_hidden_mem
    ).to(device)
    model = ContinousThoughtMachine(
        n_ticks = args.num_ticks,
        d_model = args.numd_model,
        d_memory = args.num_memory,
        n_sync_out = args.num_sync_out,
        n_sync_action = args.num_sync_action,
        d_input = args.numd_input,
        n_heads = args.num_heads,
        d_output = args.numd_output, # for both MNIST & CIFAR10
        backbone = backbone,
        neuron_lvl_model = nlm,
        synapse_model = synapses,
        dropout = args.dropout,
        neuron_selection_type = args.selection_type,
        n_random_pairing_self = args.num_rand_self_pair,
    ).to(device)
    print("Overall Model Parameters: ")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad = {param.requires_grad}")

    # --- Save Model Configuration ---
    config_path = save_model_config(run_dir, model, args)
    print(f"Model configuration saved at: {config_path}")
    
    # --- Train Model ---
    print("Starting training...")
    model, (train_losses_list, least_loss_tick_list, most_certain_tick_list) = train_classifier(
        model=model, 
        train_loader=trainloader, 
        test_loader=testloader, 
        num_iteration=args.iterations, 
        learning_rate=args.lr, 
        device=device
    )
    print("--- Training Complete ---")
    
    # --- Save Training Results ---
    model_path, full_model_path, metrics_path, final_model_info_path, model_full_backbone_path = save_training_results(
        run_dir, model, train_losses_list, least_loss_tick_list, most_certain_tick_list
    )
    print(f"Model Full Backbone saved at: {model_full_backbone_path}")
    print(f"Model weights saved at: {model_path}")
    print(f"Full model saved at: {full_model_path}")
    print(f"Training metrics saved at: {metrics_path}")
    print(f"Final model info saved at: {final_model_info_path}")
    
    # --- Create and Save Plots ---
    plots_created = create_plots(run_dir, train_losses_list, least_loss_tick_list, most_certain_tick_list)
    if plots_created:
        print("Plots created:")
        for plot_path in plots_created:
            print(f"  - {plot_path}")
    else:
        print("No plots created (empty training losses)")
    
    # --- Save Run Summary ---
    summary_path = save_summary(model_full_backbone_path, args.message, run_dir, config_path, model_path, full_model_path, metrics_path, final_model_info_path, plots_created)
    print(f"Run summary saved at: {summary_path}")
    
    print(f"\n{'='*60}")
    print(f"TRAINING RUN COMPLETE")
    print(f"{'='*60}")
    print(f"All files saved in: {run_dir}")
    print(f"DATASET: {args.name} | Ticks: {args.num_ticks} | Memory: {args.num_memory}")
    print(f"Iterations: {args.iterations} | Learning Rate: {args.lr}")
    if train_losses_list:
        print(f"Final Loss: {train_losses_list[-1]:.6f}")
        print(f"Best Loss: {min(train_losses_list):.6f}")
    print(f"{'='*60}")
