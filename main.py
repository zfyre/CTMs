import torch
import argparse
from torchvision import datasets, transforms
from models.ctm import ContinousThoughtMachine
from models.mnist import BackBone, NLM, Synapses
from train import train


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
        ])
        train_data = datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root=path, train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, num_workers=1)
        testloader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True, num_workers=1, drop_last=False)
        return trainloader, testloader
    else:
        raise ValueError(f"Dataset {name} not found")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="MNIST")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--path", type=str, default="./data")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)

    # --- Set Seed ---
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    # --- Prepare Data ---
    trainloader, testloader = prepare_data(args.name, args.batch_size, args.path)

    # --- Prepare Model ---
    model = ContinousThoughtMachine(
        n_ticks = 30,
        d_model = 128,
        d_memory = 15,
        n_sync_out = 64,
        n_sync_action = 64,
        d_input = 32,
        n_heads = 2,
        d_output = 10,

        backbone = BackBone,
        neuron_lvl_model = NLM,
        synapse_model = Synapses,

        dropout = 0,
        neuron_selection_type = 'first-last',
        n_random_pairing_self = 0,
    ).to(device)

    # --- Train Model ---
    model = train(model=model, train_loader=trainloader, test_loader=testloader, num_iteration=args.iterations, learning_rate=args.lr, device=device)
    print("--- Training Complete ---")
    # --- Save Model ---
    torch.save(model.state_dict(), f"{args.path}/ctm_{args.name.lower()}.pth")
    print(f"Model saved at {args.path}/ctm_{args.name.lower()}")

