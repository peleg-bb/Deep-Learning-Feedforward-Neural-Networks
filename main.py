# based on https://github.com/milindmalshe/Fully-Connected-Neural-Network-PyTorch/blob/master/FCN_MNIST_Classification_PyTorch.py
#
# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# import numpy as np
# import random # For setting Python's random seed
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader, random_split, Subset  # Subset for re-using full dataset
# from tabulate import tabulate  # For creating the results table
# import time  # To time the grid search
#
# # Assuming previous code (imports, NeuralNet, calculate_metrics, data splitting) is above
# # ... (all code from Task 1, 2, 3, 4, or necessary parts if run standalone) ...

# Ensure necessary imports if running this section standalone or after restarting kernel
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
from tabulate import tabulate

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import DataLoader, random_split
from sklearn.manifold import TSNE  # For t-SNE
import time


# --- Seeding Function (same as before) ---
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
# print(f"Using device: {device}") # Already printed earlier

# --- Hyperparameters for the model to be analyzed in Task 5 ---
# (These should ideally be the best ones from Task 4, or good defaults)
task5_hidden_size = 500
task5_batch_size = 64  # Batch size for training this model
task5_learning_rate = 0.001
task5_num_epochs = 5  # Train for a reasonable number of epochs
task5_seed = 42  # Seed for training this specific model

# --- MNIST Dataset Loading and Splitting (if not already done) ---
# This assumes train_dataset_new, val_dataset, test_dataset_original are available
# If running this task standalone, you'd need to redefine them.
# For continuity, we assume they are defined from previous tasks.
print(f"\n--- Task 5: Feature Analysis using t-SNE ---")
if 'train_dataset_new' not in globals():
    print("Reloading and splitting data for Task 5...")
    transform_ops = transforms.Compose([transforms.ToTensor()])
    full_train_dataset_original = torchvision.datasets.MNIST(root='./data/', train=True, transform=transform_ops,
                                                             download=True)
    initial_data_split_seed = 42  # Must be same as used before
    validation_split_size = 10000
    num_train_original = len(full_train_dataset_original)
    num_val = validation_split_size
    num_train_new_len = num_train_original - num_val
    split_generator = torch.Generator().manual_seed(initial_data_split_seed)
    train_dataset_new, val_dataset = random_split(full_train_dataset_original, [num_train_new_len, num_val],
                                                  generator=split_generator)
    test_dataset_original = torchvision.datasets.MNIST(root='./data/', train=False, transform=transform_ops,
                                                       download=True)
    print(f"New training set size: {len(train_dataset_new)}")
    print(f"Validation set size: {len(val_dataset)}")


# --- Neural Network with Feature Extraction Method ---
class NeuralNetWithFeatures(nn.Module):
    def __init__(self, current_input_size, current_hidden_size, current_num_classes):
        super(NeuralNetWithFeatures, self).__init__()
        self.input_size = current_input_size  # Store input_size for reshaping in get_hidden_features
        self.fc1 = nn.Linear(current_input_size, current_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(current_hidden_size, current_num_classes)

    def forward(self, x):
        # x is expected to be already flattened
        hidden_activated = self.relu(self.fc1(x))
        out = self.fc2(hidden_activated)
        return out

    def get_hidden_features(self, x_raw_image):
        # x_raw_image is a single image or a batch of raw images (e.g., [C, H, W] or [B, C, H, W])
        # Reshape to [B, input_size]
        x_reshaped = x_raw_image.reshape(-1, self.input_size)
        hidden_linear_output = self.fc1(x_reshaped)
        activated_hidden_output = self.relu(hidden_linear_output)  # These are the z_i
        return activated_hidden_output


# --- 1. Train a "Good" Model for Feature Analysis ---
print(
    f"\nTraining a model for Task 5 with config: HS={task5_hidden_size}, BS={task5_batch_size}, LR={task5_learning_rate}")
set_seed(task5_seed)

# Define input_size here as it's used by NeuralNetWithFeatures
mnist_input_size = 28 * 28  # Original MNIST image size flattened

model_for_tsne = NeuralNetWithFeatures(mnist_input_size, task5_hidden_size, 10).to(device)  # 10 classes
criterion_tsne = nn.CrossEntropyLoss()
optimizer_tsne = torch.optim.Adam(model_for_tsne.parameters(), lr=task5_learning_rate)


# DataLoaders for training this model
# Helper for DataLoader seeding
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


train_loader_tsne = DataLoader(dataset=train_dataset_new, batch_size=task5_batch_size, shuffle=True,
                               worker_init_fn=seed_worker if device.type != 'mps' else None,
                               generator=torch.Generator().manual_seed(task5_seed) if device.type != 'mps' else None)
val_loader_tsne = DataLoader(dataset=val_dataset, batch_size=task5_batch_size, shuffle=False)

for epoch in range(task5_num_epochs):
    model_for_tsne.train()
    epoch_loss = 0
    for i, (images, labels) in enumerate(train_loader_tsne):
        images_flat = images.reshape(-1, mnist_input_size).to(device)  # Flatten images
        labels_dev = labels.to(device)

        outputs = model_for_tsne(images_flat)
        loss = criterion_tsne(outputs, labels_dev)
        epoch_loss += loss.item() * images.size(0)

        optimizer_tsne.zero_grad()
        loss.backward()
        optimizer_tsne.step()

    avg_epoch_loss = epoch_loss / len(train_loader_tsne.dataset)
    # Simple validation error check (optional for t-SNE part, but good practice)
    model_for_tsne.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images_val, labels_val in val_loader_tsne:
            images_val_flat = images_val.reshape(-1, mnist_input_size).to(device)
            labels_val_dev = labels_val.to(device)
            outputs_val = model_for_tsne(images_val_flat)
            _, predicted_val = torch.max(outputs_val.data, 1)
            val_total += labels_val_dev.size(0)
            val_correct += (predicted_val == labels_val_dev).sum().item()
    val_acc = val_correct / val_total
    print(f"Epoch [{epoch + 1}/{task5_num_epochs}], Train Loss: {avg_epoch_loss:.4f}, Val Acc: {val_acc:.4f}")

print("Model training for t-SNE finished.")

# --- 2. Extract Features (z_i) and Raw Inputs (x_i) ---
print("\nExtracting features for t-SNE...")
model_for_tsne.eval()  # Ensure model is in evaluation mode

# We'll use all samples from train_dataset_new as requested
# If this is too slow, consider num_tsne_samples parameter.
# num_tsne_samples = len(train_dataset_new) # Use all
num_tsne_samples = min(len(train_dataset_new), 10000)  # Capping at 10k for practicality
print(f"Using {num_tsne_samples} samples for t-SNE from the training set.")

# Create a loader for the subset of train_dataset_new if not using all
# Or iterate directly through train_dataset_new (can be slow if not batched for feature extraction)
# For efficient feature extraction, use a DataLoader
feature_extraction_loader = DataLoader(dataset=train_dataset_new, batch_size=256, shuffle=False)  # No shuffle

all_raw_x = []
all_hidden_z = []
all_labels = []

count = 0
with torch.no_grad():
    for images, labels_batch in feature_extraction_loader:
        if count >= num_tsne_samples:
            break

        # For raw x_i (flattened)
        raw_x_batch = images.reshape(images.size(0), -1)  # Flatten: [B, 784]

        # For hidden z_i
        # Note: model_for_tsne.get_hidden_features expects raw image tensor, not yet flattened by loader
        # but our NeuralNetWithFeatures flattens it.
        # So, we pass images.to(device) which is [B, 1, 28, 28]
        hidden_z_batch = model_for_tsne.get_hidden_features(images.to(device))  # Pass [B, C, H, W]

        # Collect up to num_tsne_samples
        samples_to_take = min(images.size(0), num_tsne_samples - count)

        all_raw_x.append(raw_x_batch[:samples_to_take].cpu().numpy())
        all_hidden_z.append(hidden_z_batch[:samples_to_take].cpu().numpy())
        all_labels.append(labels_batch[:samples_to_take].cpu().numpy())

        count += samples_to_take

all_raw_x_np = np.concatenate(all_raw_x, axis=0)
all_hidden_z_np = np.concatenate(all_hidden_z, axis=0)
all_labels_np = np.concatenate(all_labels, axis=0)

print(f"Shape of raw inputs (x_i) for t-SNE: {all_raw_x_np.shape}")
print(f"Shape of hidden features (z_i) for t-SNE: {all_hidden_z_np.shape}")
print(f"Shape of labels for t-SNE: {all_labels_np.shape}")

# --- 3. Perform t-SNE ---
tsne_perplexity = 30
tsne_n_iter = 350  # Min 250, default 1000. 300-500 is often a good balance.
tsne_random_state = 42  # For reproducibility of t-SNE itself

print(f"\nPerforming t-SNE on raw inputs (x_i)... (perplexity={tsne_perplexity}, n_iter={tsne_n_iter})")
tsne_raw = TSNE(n_components=2, perplexity=tsne_perplexity, n_iter=tsne_n_iter, random_state=tsne_random_state,
                verbose=1)
tsne_raw_embeddings = tsne_raw.fit_transform(all_raw_x_np)
print("t-SNE on raw inputs finished.")

print(f"\nPerforming t-SNE on hidden features (z_i)... (perplexity={tsne_perplexity}, n_iter={tsne_n_iter})")
tsne_hidden = TSNE(n_components=2, perplexity=tsne_perplexity, n_iter=tsne_n_iter, random_state=tsne_random_state,
                   verbose=1)
tsne_hidden_embeddings = tsne_hidden.fit_transform(all_hidden_z_np)
print("t-SNE on hidden features finished.")


# --- 4. Plot t-SNE Embeddings ---
def plot_tsne_embeddings(embeddings, labels, title, num_classes=10):
    plt.figure(figsize=(10, 8))
    # Get a colormap for 10 distinct colors
    try:
        cmap = plt.cm.get_cmap('tab10', num_classes)
    except AttributeError:  # older matplotlib
        cmap = plt.cm.tab10

    for i in range(num_classes):
        # Select data for the i-th class
        idx = (labels == i)
        plt.scatter(embeddings[idx, 0], embeddings[idx, 1], color=cmap(i), label=str(i), s=10, alpha=0.7)

    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(markerscale=2, title="Digits")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


plot_tsne_embeddings(tsne_raw_embeddings, all_labels_np, "t-SNE visualization of raw MNIST pixels (x_i)")
plot_tsne_embeddings(tsne_hidden_embeddings, all_labels_np,
                     f"t-SNE visualization of hidden layer features (z_i) - HS={task5_hidden_size}")

print("\n--- Task 5 Analysis Questions (to be answered based on the plots): ---")
print("1. What can you say about the differences between these plots (raw x_i vs. hidden z_i)?")
print(
    "   Consider: How well are the classes (digits) separated? Are there clear clusters? How compact are the clusters?")
print("2. What can you say about the learned model based on the t-SNE of z_i?")
print(
    "   Consider: Does the first layer seem to be learning features that help distinguish between digits? How does this relate to the model's purpose (classification)?")
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