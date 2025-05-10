# based on https://github.com/milindmalshe/Fully-Connected-Neural-Network-PyTorch/blob/master/FCN_MNIST_Classification_PyTorch.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np  # For plotting misclassified images

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001
num_misclassified_to_show = 10

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data/',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# --- Optional: Plot some training images (from original code) ---
images_sample, labels_sample = next(iter(train_loader))
fig_sample, axs_sample = plt.subplots(2, 5, figsize=(10, 4))
fig_sample.suptitle('Sample MNIST Training Images')
for ii in range(2):
    for jj in range(5):
        idx = 5 * ii + jj
        axs_sample[ii, jj].imshow(images_sample[idx].squeeze(), cmap='gray')
        axs_sample[ii, jj].set_title(f"Label: {labels_sample[idx].item()}")
        axs_sample[ii, jj].axis('off')
plt.tight_layout()
plt.show()
# --- End Optional Plot ---

# Fully connected neural network
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


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Function to calculate loss and error
def calculate_metrics(loader, model, criterion, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)  # Accumulate weighted loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    error = 1 - accuracy
    return avg_loss, error


# Lists to store metrics for plotting
train_losses_epoch = []
train_errors_epoch = []
test_errors_epoch = []

# Train the model
total_step = len(train_loader)
print(f"Starting training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_batch_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_batch_loss += loss.item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 200 == 0:  # Print training progress more frequently
            print('Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # Calculate metrics after each epoch
    avg_epoch_train_loss_on_batches = running_batch_loss / total_step  # Avg loss from training batches

    # For e_tr (train error) and comprehensive train loss, evaluate on the whole training set
    epoch_train_loss, epoch_train_error = calculate_metrics(train_loader, model, criterion, device)
    train_losses_epoch.append(epoch_train_loss)  # Using loss from full train set eval
    train_errors_epoch.append(epoch_train_error)

    # For e_te (test error)
    _, epoch_test_error = calculate_metrics(test_loader, model, criterion, device)
    test_errors_epoch.append(epoch_test_error)

    print(f"Epoch [{epoch + 1}/{num_epochs}]: "
          f"Train Loss (full eval): {epoch_train_loss:.4f}, Train Error (e_tr): {epoch_train_error:.4f}, "
          f"Test Error (e_te): {epoch_test_error:.4f}")

print("Training finished.")

# --- Final Evaluation and Reporting ---
print("\n--- Final Model Evaluation ---")
final_test_loss, final_test_error = calculate_metrics(test_loader, model, criterion, device)
final_test_accuracy = 1 - final_test_error

print(f'Final Test Accuracy on the 10000 test images: {final_test_accuracy * 100:.2f} %')
print(f'Final Test Error (e_te) on the 10000 test images: {final_test_error:.4f}')

# --- Collect Misclassified Images ---
misclassified_images_list = []
misclassified_true_labels = []
misclassified_pred_labels = []

model.eval()  # Ensure model is in evaluation mode
with torch.no_grad():
    for images, labels in test_loader:
        original_images_batch = images.clone()  # Keep original shape for plotting
        images_reshaped = images.reshape(-1, input_size).to(device)
        labels_dev = labels.to(device)
        outputs = model(images_reshaped)
        _, predicted = torch.max(outputs.data, 1)

        for j in range(labels_dev.size(0)):
            if predicted[j] != labels_dev[j]:
                if len(misclassified_images_list) < num_misclassified_to_show:
                    misclassified_images_list.append(original_images_batch[j].cpu())
                    misclassified_true_labels.append(labels_dev[j].cpu().item())
                    misclassified_pred_labels.append(predicted[j].cpu().item())
            # else: # Optimization: stop collecting if we have enough
            #     if len(misclassified_images_list) >= num_misclassified_to_show and \
            #        all(len(lst) >= num_misclassified_to_show for lst in [misclassified_images_list, misclassified_true_labels, misclassified_pred_labels]):
            #         break # break inner loop
    # if len(misclassified_images_list) >= num_misclassified_to_show and \
    #    all(len(lst) >= num_misclassified_to_show for lst in [misclassified_images_list, misclassified_true_labels, misclassified_pred_labels]):
    #     break # break outer loop (data loader)

# --- Plotting Section ---

# 1. Plot Train and Test Error graphs
plt.figure(figsize=(10, 5))
epochs_range = range(1, num_epochs + 1)
plt.plot(epochs_range, train_errors_epoch, label='Train Error (e_tr)', marker='o')
plt.plot(epochs_range, test_errors_epoch, label='Test Error (e_te)', marker='x')
plt.title('Train and Test Error vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Error Rate')
plt.xticks(epochs_range)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Plot Training Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_losses_epoch, label='Average Train Loss (full eval)', marker='s')
plt.title('Average Training Loss (full eval) vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(epochs_range)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Plot Misclassified Images
if misclassified_images_list:
    print(f"\n--- Plotting {len(misclassified_images_list)} Misclassified Images ---")
    num_cols = 5
    num_rows = (len(misclassified_images_list) + num_cols - 1) // num_cols  # Calculate rows needed
    fig_mis, axs_mis = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2.5, num_rows * 3))
    axs_mis = axs_mis.flatten()  # Flatten in case of single row/col

    for i in range(len(misclassified_images_list)):
        img = misclassified_images_list[i].squeeze()
        true_label = misclassified_true_labels[i]
        pred_label = misclassified_pred_labels[i]
        axs_mis[i].imshow(img, cmap='gray')
        axs_mis[i].set_title(f"True: {true_label}\nPred: {pred_label}")
        axs_mis[i].axis('off')

    # Turn off any unused subplots
    for i in range(len(misclassified_images_list), len(axs_mis)):
        axs_mis[i].axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("\nNo misclassified images to show (or none were collected).")

# Save the model checkpoint (optional, from original code)
# torch.save(model.state_dict(), 'model.ckpt')