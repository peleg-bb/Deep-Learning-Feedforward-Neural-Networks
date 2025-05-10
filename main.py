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


# --- Seeding Function ---
def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # The following two lines are crucial for reproducibility with CUDA
        # but can impact performance. For this task, reproducibility is key.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed}")

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters (same as Task 1)
input_size = 784  # 28x28
hidden_size = 500
num_classes = 10
num_epochs = 5 # As per original, can be increased for better convergence
batch_size = 100
learning_rate = 0.001

# --- MNIST dataset (loaded once) ---
# Ensure transforms.ToTensor() is applied, which normalizes images to [0,1]
transform_ops = transforms.Compose([
    transforms.ToTensor()
    # No other normalization like (mean, std) was in the original,
    # so keeping it simple. Adding it might improve performance slightly.
])

train_dataset = torchvision.datasets.MNIST(root='./data/',
                                           train=True,
                                           transform=transform_ops,
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data/',
                                          train=False,
                                          transform=transform_ops)

# Fully connected neural network (same as Task 1)
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

# Function to calculate loss and error (same as Task 1, slightly refined)
def calculate_metrics(loader, model, criterion, device, is_train=False):
    if is_train:
        model.train() # Keep model in train mode if calculating training metrics during training phase
    else:
        model.eval()  # Set model to evaluation mode for test/validation

    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad() if not is_train else torch.enable_grad(): # No grad for eval
        for images, labels in loader:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0) # Accumulate weighted loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    error = 1 - accuracy
    return avg_loss, error


# --- Main Experiment Loop for Multiple Seeds ---
seed_numbers = [1992, 2006, 2009, 2011, 2015] # 5 seeds which represent the years Barca won the Champions League
all_runs_test_errors_epoch = [] # To store [[epoch_errors_run1], [epoch_errors_run2], ...]
final_test_errors_for_runs = [] # To store [final_error_run1, final_error_run2, ...]

for run_idx, current_seed in enumerate(seed_numbers):
    print(f"\n--- Starting Run {run_idx + 1}/{len(seed_numbers)} with Seed: {current_seed} ---")
    set_seed(current_seed)

    # --- Re-initialize Model and Optimizer for each run ---
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --- Re-initialize DataLoaders to ensure shuffle is re-seeded if applicable ---
    # The shuffle in train_loader is controlled by torch.manual_seed()
    # A worker_init_fn would be needed for num_workers > 0 for full determinism,
    # but for num_workers=0 (default), global seed is usually sufficient.
    def seed_worker(worker_id): # Added for completeness if num_workers > 0
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               worker_init_fn=seed_worker if device.type != 'mps' else None, # MPS doesn't support worker_init_fn well
                                               generator=torch.Generator().manual_seed(current_seed) if device.type != 'mps' else None) # For DataLoader shuffle

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False) # No shuffle, so less critical but good practice

    # Lists to store metrics for the current run
    current_run_train_errors_epoch = []
    current_run_test_errors_epoch = []

    print(f"Training model for {num_epochs} epochs with seed {current_seed}...")
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        running_batch_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_batch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 200 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        # Calculate metrics after each epoch
        # For e_tr (train error), evaluate on the whole training set
        # Pass is_train=False to ensure no_grad context if calculate_metrics uses it
        _, epoch_train_error = calculate_metrics(train_loader, model, criterion, device, is_train=False)
        current_run_train_errors_epoch.append(epoch_train_error)

        # For e_te (test error)
        _, epoch_test_error = calculate_metrics(test_loader, model, criterion, device, is_train=False)
        current_run_test_errors_epoch.append(epoch_test_error)

        print(f"Epoch [{epoch+1}/{num_epochs}] (Seed {current_seed}): "
              f"Train Error (e_tr): {epoch_train_error:.4f}, "
              f"Test Error (e_te): {epoch_test_error:.4f}")

    all_runs_test_errors_epoch.append(current_run_test_errors_epoch)
    final_test_errors_for_runs.append(current_run_test_errors_epoch[-1]) # Last epoch's test error
    print(f"Run {run_idx + 1} (Seed {current_seed}) Final Test Error: {current_run_test_errors_epoch[-1]:.4f}")

print("\n--- All Runs Completed ---")

# --- Plotting Test Errors from all runs ---
plt.figure(figsize=(12, 7))
epochs_range = range(1, num_epochs + 1)
for i, run_errors in enumerate(all_runs_test_errors_epoch):
    plt.plot(epochs_range, run_errors, label=f'Run {i+1} (Seed {seed_numbers[i]})', marker='o', linestyle='-')

plt.title('Test Error (e_te) vs. Epochs for Different Seeds')
plt.xlabel('Epoch')
plt.ylabel('Test Error Rate')
plt.xticks(epochs_range)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Calculate and Report Mean and Standard Deviation of Final Test Errors ---
mean_final_test_error = np.mean(final_test_errors_for_runs)
std_final_test_error = np.std(final_test_errors_for_runs)
variance_final_test_error = np.var(final_test_errors_for_runs) # Variance is std^2

print("\n--- Statistics of Final Test Errors Across Runs ---")
print(f"Final Test Errors for each run: {[f'{err:.4f}' for err in final_test_errors_for_runs]}")
print(f"Mean Final Test Error: {mean_final_test_error:.4f}")
print(f"Standard Deviation of Final Test Error: {std_final_test_error:.4f}")
print(f"Variance of Final Test Error: {variance_final_test_error:.6f}") # More precision for variance

# --- Assess Robustness ---
print("\n--- Robustness Assessment ---")
if std_final_test_error < 0.005: # Threshold for "small" std dev (e.g., < 0.5% error difference)
    print("The model appears to be reasonably robust to the choice of seed number.")
    print("The standard deviation of the final test errors is small, indicating consistent performance across different initializations.")
elif std_final_test_error < 0.01:
    print("The model shows moderate sensitivity to the choice of seed number.")
    print("There is some variation in performance, but it's not excessively large.")
else:
    print("The model appears to be sensitive to the choice of seed number.")
    print("The standard deviation of the final test errors is relatively large, indicating that results can vary noticeably with different initializations.")
print("Note: A small number of epochs (5) might lead to higher variance as the model may not have fully converged.")
print("Longer training could potentially reduce this variance.")# torch.save(model.state_dict(), 'model.ckpt')