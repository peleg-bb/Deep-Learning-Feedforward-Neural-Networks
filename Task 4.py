# based on https://github.com/milindmalshe/Fully-Connected-Neural-Network-PyTorch/blob/master/FCN_MNIST_Classification_PyTorch.py

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random # For setting Python's random seed
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset  # Subset for re-using full dataset
from tabulate import tabulate  # For creating the results table
import time  # To time the grid search


# --- Seeding Function (same as before) ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # print(f"Seed set to: {seed}") # Can be verbose for grid search


# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Fixed Hyperparameters (not part of grid search)
input_size = 784  # MNIST specific
num_classes = 10  # MNIST specific
num_epochs_grid_search = 5  # Reduced epochs for grid search to save time, adjust as needed
# For a thorough search, 15-20 might be better, but will take longer.
validation_split_size = 10000
initial_data_split_seed = 42  # Seed for the initial train/validation split
grid_search_run_seed = 777  # Single seed for all training runs within the grid search

# --- MNIST Dataset Loading and Splitting (Done ONCE) ---
transform_ops = transforms.Compose([transforms.ToTensor()])
full_train_dataset_original = torchvision.datasets.MNIST(root='./data/', train=True, transform=transform_ops,
                                                         download=True)
test_dataset_original = torchvision.datasets.MNIST(root='./data/', train=False, transform=transform_ops, download=True)

num_train_original = len(full_train_dataset_original)
num_val = validation_split_size
num_train_new = num_train_original - num_val
split_generator = torch.Generator().manual_seed(initial_data_split_seed)
train_dataset_new, val_dataset = random_split(full_train_dataset_original, [num_train_new, num_val],
                                              generator=split_generator)

print(f"New training set size: {len(train_dataset_new)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset_original)}")

# Test DataLoader (created once as batch_size for test/val doesn't change grid params)
# Using a fixed batch size for validation and test for consistency during evaluation
eval_batch_size = 256  # Can be larger for evaluation
val_loader_fixed_bs = DataLoader(dataset=val_dataset, batch_size=eval_batch_size, shuffle=False)
test_loader_fixed_bs = DataLoader(dataset=test_dataset_original, batch_size=eval_batch_size, shuffle=False)


# Neural Network Definition (same as before)
class NeuralNet(nn.Module):
    def __init__(self, current_input_size, current_hidden_size, current_num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(current_input_size, current_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(current_hidden_size, current_num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Function to calculate metrics (same as before)
def calculate_metrics(loader, model, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images_reshaped = images.reshape(-1, input_size).to(device)  # Ensure correct input_size
            labels_dev = labels.to(device)
            outputs = model(images_reshaped)
            loss = criterion(outputs, labels_dev)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels_dev.size(0)
            correct += (predicted == labels_dev).sum().item()
    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    error = 1 - accuracy
    return avg_loss, error


# --- Grid Search Definition ---
param_grid = {
    'hidden_size': [200, 500],  # Reduced set for quicker demo
    'batch_size': [64, 128],  # Reduced set
    'learning_rate': [0.001, 0.0005]  # Reduced set
}
# For a more thorough search, you might use:
# 'hidden_size': [128, 256, 512],
# 'batch_size': [32, 64, 128, 256],
# 'learning_rate': [0.01, 0.005, 0.001, 0.0005, 0.0001]

results_data = []
best_overall_eva_star = float('inf')
best_params_config = None
best_ete_star_for_best_eva = float('inf')

total_combinations = len(param_grid['hidden_size']) * len(param_grid['batch_size']) * len(param_grid['learning_rate'])
current_combination = 0

print(f"\n--- Starting Grid Search ({total_combinations} combinations) ---")
grid_search_start_time = time.time()


# Helper for DataLoader seeding
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


for hs_val in param_grid['hidden_size']:
    for bs_val in param_grid['batch_size']:
        for lr_val in param_grid['learning_rate']:
            current_combination += 1
            current_config = {'hidden_size': hs_val, 'batch_size': bs_val, 'learning_rate': lr_val}
            print(f"\nCombination {current_combination}/{total_combinations}: {current_config}")

            # --- Set seed for this specific training run for reproducibility ---
            set_seed(grid_search_run_seed)

            # --- Initialize Model, Optimizer, and Train DataLoader for current config ---
            model = NeuralNet(input_size, hs_val, num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_val)

            # Train DataLoader uses the current batch_size from the grid
            train_loader_current_bs = DataLoader(dataset=train_dataset_new, batch_size=bs_val, shuffle=True,
                                                 worker_init_fn=seed_worker if device.type != 'mps' else None,
                                                 generator=torch.Generator().manual_seed(
                                                     grid_search_run_seed) if device.type != 'mps' else None)

            min_eva_this_config = float('inf')
            ete_at_min_eva_this_config = float('inf')

            # --- Training Loop for current config ---
            total_steps_train = len(train_loader_current_bs)
            for epoch in range(num_epochs_grid_search):
                model.train()
                epoch_start_time = time.time()
                for i, (images, labels) in enumerate(train_loader_current_bs):
                    images = images.reshape(-1, input_size).to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # --- Evaluation after epoch ---
                # model.eval() is handled inside calculate_metrics
                _, e_tr_epoch = calculate_metrics(train_loader_current_bs, model, criterion,
                                                  device)  # Train error on current train loader
                _, e_va_epoch = calculate_metrics(val_loader_fixed_bs, model, criterion, device)
                _, e_te_epoch = calculate_metrics(test_loader_fixed_bs, model, criterion, device)

                # print(f"  Epoch [{epoch+1}/{num_epochs_grid_search}]: e_tr: {e_tr_epoch:.4f}, e_va: {e_va_epoch:.4f}, e_te: {e_te_epoch:.4f} ({time.time() - epoch_start_time:.2f}s)")

                if e_va_epoch < min_eva_this_config:
                    min_eva_this_config = e_va_epoch
                    ete_at_min_eva_this_config = e_te_epoch

            print(
                f"  Finished training for {current_config}. Best e_va*: {min_eva_this_config:.4f}, Corresponding e_te*: {ete_at_min_eva_this_config:.4f}")

            results_data.append({
                'Hidden Size': hs_val,
                'Batch Size': bs_val,
                'Learning Rate': lr_val,
                'e_va* (Best Val Error)': min_eva_this_config,
                'e_te* (at Best Val Error)': ete_at_min_eva_this_config
            })

            # Check if this is the best overall configuration based on e_va*
            if min_eva_this_config < best_overall_eva_star:
                best_overall_eva_star = min_eva_this_config
                best_ete_star_for_best_eva = ete_at_min_eva_this_config
                best_params_config = current_config
            # Tie-breaking: if e_va* is the same, prefer lower e_te*
            elif min_eva_this_config == best_overall_eva_star and ete_at_min_eva_this_config < best_ete_star_for_best_eva:
                best_ete_star_for_best_eva = ete_at_min_eva_this_config
                best_params_config = current_config

grid_search_duration = time.time() - grid_search_start_time
print(f"\n--- Grid Search Completed in {grid_search_duration:.2f} seconds ---")

# --- Display Results Table ---
headers = ["Hidden Size", "Batch Size", "Learning Rate", "e_va* (Best Val Error)", "e_te* (at Best Val Error)"]
table_data = []
for r in results_data:
    is_best = (r['Hidden Size'] == best_params_config['hidden_size'] and
               r['Batch Size'] == best_params_config['batch_size'] and
               r['Learning Rate'] == best_params_config['learning_rate'])

    row = [
        r['Hidden Size'],
        r['Batch Size'],
        f"{r['Learning Rate']:.4f}",  # Format LR
        f"{r['e_va* (Best Val Error)']:.4f}",
        f"{r['e_te* (at Best Val Error)']:.4f}"
    ]
    if is_best:  # Highlight the best row
        row = [f"**{val}**" for val in row]  # Markdown-style bold
    table_data.append(row)

print("\n--- Grid Search Results ---")
print(tabulate(table_data, headers=headers, tablefmt="pipe"))  # "pipe" is good for markdown

print("\n--- Best Hyperparameter Combination ---")
if best_params_config:
    print(f"Parameters: {best_params_config}")
    print(f"Achieved Best Validation Error (e_va*): {best_overall_eva_star:.4f}")
    print(f"Corresponding Test Error (e_te*): {best_ete_star_for_best_eva:.4f}")
else:
    print("No best configuration found (this should not happen if grid search ran).")