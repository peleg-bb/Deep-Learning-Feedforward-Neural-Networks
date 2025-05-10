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


# --- Seeding Function (same as Task 2) ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed}")


# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 15  # Increased epochs slightly to better observe validation behavior
batch_size = 100
learning_rate = 0.001
validation_split_size = 10000
initial_data_split_seed = 42  # Seed for the initial train/validation split

# --- MNIST Dataset Loading and Splitting ---
transform_ops = transforms.Compose([transforms.ToTensor()])

# Load the full original training dataset
full_train_dataset_original = torchvision.datasets.MNIST(root='./data/',
                                                         train=True,
                                                         transform=transform_ops,
                                                         download=True)

# Load the original test dataset
test_dataset_original = torchvision.datasets.MNIST(root='./data/',
                                                   train=False,
                                                   transform=transform_ops,
                                                   download=True)

# Split the original training set into a new training set and a validation set
num_train_original = len(full_train_dataset_original)
num_val = validation_split_size
num_train_new = num_train_original - num_val

# Use a fixed generator for the initial split for consistency across script reruns
split_generator = torch.Generator().manual_seed(initial_data_split_seed)
train_dataset_new, val_dataset = random_split(full_train_dataset_original,
                                              [num_train_new, num_val],
                                              generator=split_generator)

print(f"Original training set size: {num_train_original}")
print(f"New training set size: {len(train_dataset_new)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset_original)}")


# Fully connected neural network (same as before)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Function to calculate loss and error (same as before)
def calculate_metrics(loader, model, criterion, device, is_train_phase=False):
    if is_train_phase:  # if called during training step for batch loss etc.
        pass  # model already in train mode
    else:
        model.eval()

    total_loss = 0
    correct = 0
    total = 0
    # For eval/test, no gradients needed. For train error calc, also no grad.
    with torch.no_grad():
        for images, labels in loader:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)  # Use criterion for loss
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if not is_train_phase:  # if called for eval, switch back to train if it was
        model.train()  # Important: switch back to train mode after eval if it was called mid-epoch cycle

    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    error = 1 - accuracy
    return avg_loss, error


# --- Main Experiment Loop for Multiple Seeds ---
seed_numbers = [1992, 2006, 2009, 2011, 2015] # 5 seeds which represent the years Barca won the Champions League
all_runs_test_errors_epoch_curves = []  # For plotting full e_te curves
all_runs_val_errors_epoch_curves = []  # For plotting full e_va curves
reported_metrics_per_run = []  # List to store (e_te*, e_va*) for each run


# Helper for DataLoader seeding
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


for run_idx, current_seed in enumerate(seed_numbers):
    print(f"\n--- Starting Run {run_idx + 1}/{len(seed_numbers)} with Seed: {current_seed} ---")
    set_seed(current_seed)

    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # DataLoaders - train_dataset_new and val_dataset are fixed subsets
    # Shuffling of train_loader and val_loader (if shuffle=True) will depend on current_seed
    train_loader = DataLoader(dataset=train_dataset_new, batch_size=batch_size, shuffle=True,
                              worker_init_fn=seed_worker if device.type != 'mps' else None,
                              generator=torch.Generator().manual_seed(current_seed) if device.type != 'mps' else None)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)  # No shuffle for validation
    test_loader = DataLoader(dataset=test_dataset_original, batch_size=batch_size, shuffle=False)  # No shuffle for test

    current_run_epoch_train_errors = []
    current_run_epoch_val_errors = []
    current_run_epoch_test_errors = []

    min_val_error_this_run = float('inf')
    test_error_at_min_val_error = float('inf')
    best_epoch_this_run = -1

    print(f"Training model for {num_epochs} epochs with seed {current_seed}...")
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % (total_step // 2) == 0:  # Print a few times per epoch
                print('Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        # Calculate errors after each epoch
        # model.eval() is handled within calculate_metrics
        _, e_tr = calculate_metrics(train_loader, model, criterion, device)
        _, e_va = calculate_metrics(val_loader, model, criterion, device)
        _, e_te = calculate_metrics(test_loader, model, criterion, device)

        current_run_epoch_train_errors.append(e_tr)
        current_run_epoch_val_errors.append(e_va)
        current_run_epoch_test_errors.append(e_te)

        print(f"Epoch [{epoch + 1}/{num_epochs}] (Seed {current_seed}): "
              f"e_tr: {e_tr:.4f}, e_va: {e_va:.4f}, e_te: {e_te:.4f}")

        # Check for new minimum validation error
        if e_va < min_val_error_this_run:
            min_val_error_this_run = e_va
            test_error_at_min_val_error = e_te  # Store e_te from this epoch
            best_epoch_this_run = epoch + 1

    all_runs_test_errors_epoch_curves.append(current_run_epoch_test_errors)
    all_runs_val_errors_epoch_curves.append(current_run_epoch_val_errors)

    # Store the (e_te*, e_va*) pair for this run
    # e_va* is min_val_error_this_run
    # e_te* is test_error_at_min_val_error
    reported_metrics_per_run.append((test_error_at_min_val_error, min_val_error_this_run))

    print(f"Run {run_idx + 1} (Seed {current_seed}) - Best Validation Error (e_va*): {min_val_error_this_run:.4f} "
          f"(at epoch {best_epoch_this_run}), "
          f"Corresponding Test Error (e_te*): {test_error_at_min_val_error:.4f}")

print("\n--- All Runs Completed ---")

# --- Reporting the (e_te*, e_va*) pairs ---
print("\n--- Reported (e_te*, e_va*) for each run ---")
for i, (ete_star, eva_star) in enumerate(reported_metrics_per_run):
    print(f"Run {i + 1} (Seed {seed_numbers[i]}): e_te* = {ete_star:.4f}, e_va* = {eva_star:.4f}")

# --- Plotting Test Errors from all runs ---
plt.figure(figsize=(12, 7))
epochs_range_plot = range(1, num_epochs + 1)
for i, run_errors in enumerate(all_runs_test_errors_epoch_curves):
    plt.plot(epochs_range_plot, run_errors, label=f'Run {i + 1} (Seed {seed_numbers[i]})', marker='.', linestyle='-')
plt.title('Test Error (e_te) vs. Epochs for Different Seeds')
plt.xlabel('Epoch')
plt.ylabel('Test Error Rate')
plt.xticks(epochs_range_plot)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plotting Validation Errors from all runs ---
plt.figure(figsize=(12, 7))
for i, run_errors in enumerate(all_runs_val_errors_epoch_curves):
    plt.plot(epochs_range_plot, run_errors, label=f'Run {i + 1} (Seed {seed_numbers[i]})', marker='.', linestyle='-')
plt.title('Validation Error (e_va) vs. Epochs for Different Seeds')
plt.xlabel('Epoch')
plt.ylabel('Validation Error Rate')
plt.xticks(epochs_range_plot)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Optional: Calculate and print mean/std of the reported e_te* and e_va* ---
reported_ete_stars = [pair[0] for pair in reported_metrics_per_run]
reported_eva_stars = [pair[1] for pair in reported_metrics_per_run]

mean_ete_star = np.mean(reported_ete_stars)
std_ete_star = np.std(reported_ete_stars)
mean_eva_star = np.mean(reported_eva_stars)
std_eva_star = np.std(reported_eva_stars)

print("\n--- Statistics of Reported Metrics Across Runs ---")
print(f"Mean e_te*: {mean_ete_star:.4f}, Std e_te*: {std_ete_star:.4f}")
print(f"Mean e_va*: {mean_eva_star:.4f}, Std e_va*: {std_eva_star:.4f}")