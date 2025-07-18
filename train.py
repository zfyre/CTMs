import torch
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm
from models.ctm import ContinousThoughtMachine
from models.ctm_dyn_kv import ContinousThoughtMachineDyn
from utils import calculate_accuracy, calculate_accuracy_mnist_count
from losses import loss_classifier_, loss_mnist_count_

def train_classifier(
        model: ContinousThoughtMachine,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_iteration: int,
        learning_rate: float,
        device: torch.device,
):
    test_every = 100
    optim = torch.optim.AdamW(model.parameters(), lr = learning_rate) # TODO: Check if list(model.parameters()) is not the cause of no accuracy or are these same

    model.train()
    with tqdm(total=num_iteration, initial=0, dynamic_ncols=True) as pbar:
        
        # Training Results across iterations
        train_losses_list = []
        least_loss_tick_list = []
        most_certain_tick_list = []


        test_loss = None
        test_accuracy = None
        for step in range(num_iteration):
            inputs, targets = next(iter(train_loader))
            # print("Inputs: ", inputs.shape)
            inputs, targets = inputs.to(device), targets.to(device) # Offloading the batch to GPU
            predictions, certainities, (decay_action, decay_out) = model(inputs, track=False)
            train_loss, (least_loss_tick, most_certain_tick) = loss_classifier_(predictions, certainities, targets)
            train_accuracy = calculate_accuracy(predictions, targets, most_certain_tick)

            optim.zero_grad()
            train_loss.backward()
            optim.step()

            if step % test_every == 0:
                model.eval()
                with torch.inference_mode():
                    all_test_predictions = []
                    all_test_targets = []
                    all_test_where_most_certain = []
                    all_test_losses = []

                    for inputs, targets in test_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        predictions, certainties, _ = model(inputs, track=False)
                        test_loss, (_, where_most_certain) = loss_classifier_(predictions, certainties, targets)
                        all_test_losses.append(test_loss.item())

                        all_test_predictions.append(predictions)
                        all_test_targets.append(targets)
                        all_test_where_most_certain.append(where_most_certain)

                    # NOTE Here by concatinating along the dim=0 we are technically increasing the batch size & hence if someway our test_loader is big the GPU might not be able to handle this request of calculating the accuracy
                    all_test_predictions = torch.cat(all_test_predictions, dim=0) 
                    all_test_targets = torch.cat(all_test_targets, dim=0)
                    all_test_where_most_certain = torch.cat(all_test_where_most_certain, dim=0)

                    test_accuracy = calculate_accuracy(all_test_predictions, all_test_targets, all_test_where_most_certain)
                    test_loss = sum(all_test_losses) / len(all_test_losses)
                model.train()
            
            train_losses_list.append(train_loss.item())
            least_loss_tick_list.append(least_loss_tick.detach().cpu().numpy())
            most_certain_tick_list.append(most_certain_tick.detach().cpu().numpy())
            
            pbar.set_description(f'Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.3f} Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}, Decay Params: {decay_action.max():.3f}, {decay_out.max():.3f}')
            pbar.update(1)

    return model, (train_losses_list, least_loss_tick_list, most_certain_tick_list)

def train_mnist_count(
        model: ContinousThoughtMachineDyn,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_iteration: int,
        learning_rate: float,
        device: torch.device,
):
    test_every = 100
    optim = torch.optim.AdamW(model.parameters(), lr = learning_rate) # TODO: Check if list(model.parameters()) is not the cause of no accuracy or are these same

    model.train()
    with tqdm(total=num_iteration, initial=0, dynamic_ncols=True) as pbar:
        
        # Training Results across iterations
        train_losses_list = []
        least_loss_tick_list = []
        most_certain_tick_list = []


        test_loss = None
        test_accuracy = None
        for step in range(num_iteration):
            inputs, targets = next(iter(train_loader))
            # print("Inputs: ", inputs.shape)
            inputs, targets = inputs.to(device), targets.to(device) # Offloading the batch to GPU
            predictions, certainities, (decay_action, decay_out) = model(inputs, track=False)
            train_loss, (least_loss_tick, most_certain_tick) = loss_mnist_count_(predictions, certainities, targets)
            train_accuracy = calculate_accuracy_mnist_count(predictions, targets, most_certain_tick)

            optim.zero_grad()
            train_loss.backward()
            optim.step()

            if step % test_every == 0:
                model.eval()
                with torch.inference_mode():
                    all_test_predictions = []
                    all_test_targets = []
                    all_test_where_most_certain = []
                    all_test_losses = []

                    for inputs, targets in test_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        predictions, certainties, _ = model(inputs, track=False)
                        test_loss, (_, where_most_certain) = loss_mnist_count_(predictions, certainties, targets)
                        all_test_losses.append(test_loss.item())

                        all_test_predictions.append(predictions)
                        all_test_targets.append(targets)
                        all_test_where_most_certain.append(where_most_certain)

                    # NOTE Here by concatinating along the dim=0 we are technically increasing the batch size & hence if someway our test_loader is big the GPU might not be able to handle this request of calculating the accuracy
                    all_test_predictions = torch.cat(all_test_predictions, dim=0) 
                    all_test_targets = torch.cat(all_test_targets, dim=0)
                    all_test_where_most_certain = torch.cat(all_test_where_most_certain, dim=0)

                    test_accuracy = calculate_accuracy_mnist_count(all_test_predictions, all_test_targets, all_test_where_most_certain)
                    test_loss = sum(all_test_losses) / len(all_test_losses)
                model.train()
            
            train_losses_list.append(train_loss.item())
            least_loss_tick_list.append(least_loss_tick.detach().cpu().numpy())
            most_certain_tick_list.append(most_certain_tick.detach().cpu().numpy())
            
            pbar.set_description(f'Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.3f} Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}, Decay Params: {decay_action.max():.3f}, {decay_out.max():.3f}')
            pbar.update(1)

    return model, (train_losses_list, least_loss_tick_list, most_certain_tick_list)
