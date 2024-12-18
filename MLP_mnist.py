import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def gradients_to_vector(grad_dict):
    """
    Converts a dictionary of gradients into a single flattened vector.

    Args:
    - grad_dict: Dictionary where keys are parameter names and values are gradient tensors.

    Returns:
    - vector: A 1D tensor of shape (1, num_params), where num_params is the total number of parameters.
    - shapes: A dictionary mapping parameter names to their original shapes (used for reconstruction).
    """
    # Flatten each gradient tensor and store its shape
    flattened_grads = []
    shapes = {}

    for name, grad in grad_dict.items():
        shapes[name] = grad.shape
        flattened_grads.append(grad.view(-1))  # Flatten the gradient tensor

    # Concatenate all flattened gradients into a single vector
    vector = torch.cat(flattened_grads).view(1, -1)
    return vector, shapes

def vector_to_gradients(vector, shapes):
    """
    Converts a flattened gradient vector back into a dictionary of gradients.

    Args:
    - vector: A 1D tensor of shape (1, num_params).
    - shapes: A dictionary mapping parameter names to their original shapes.

    Returns:
    - grad_dict: Dictionary where keys are parameter names and values are gradient tensors reshaped to original shapes.
    """
    grad_dict = {}
    start = 0  # Start index for slicing the vector

    for name, shape in shapes.items():
        # Calculate the number of elements for this parameter
        num_elements = torch.prod(torch.tensor(shape)).item()

        # Extract and reshape the corresponding part of the vector
        grad_dict[name] = vector[0, start:start+num_elements].view(shape)

        # Update start index
        start += num_elements

    return grad_dict

def loss_from_vector(dataset, param_vector, model, criterion):
    """
    Calculates the loss over a dataset by setting the model parameters to the values
    provided in the flattened parameter vector.

    Args:
    - model: The neural network model (e.g., SimpleNN).
    - criterion: The loss function (e.g., CrossEntropyLoss).
    - dataloader: DataLoader for the dataset.
    - param_vector: A 1D tensor containing the flattened model parameters.

    Returns:
    - loss: The loss calculated over the dataset.
    """
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    shapes = {name: param.shape for name, param in model.named_parameters()}

    grad_dict = vector_to_gradients(param_vector, shapes)

    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(grad_dict[name])

    #model.to(device)
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            #images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)

    # Compute the average loss
    average_loss = total_loss / total_samples
    return average_loss

def gradient_from_vector(dataset, param_vector, model, criterion, batch_size=None):
    """
    Calculate gradients of all parameters of the model over an entire dataset.

    Args:
    - model: The neural network model (e.g., SimpleNN).
    - criterion: The loss function (e.g., CrossEntropyLoss).
    - dataloader: DataLoader for the dataset.

    Returns:
    - avg_gradients: Dictionary containing the average gradients for all parameters.
    """
    if batch_size is None:
        batch_size = len(dataset)

    indices = list(range(len(dataset)))
    random_indices = random.sample(indices, batch_size)
    subset = Subset(dataset, random_indices)
    dataloader = DataLoader(subset, batch_size=64, shuffle=True)

    #model.to(device)
    model.eval()

    shapes = {name: param.shape for name, param in model.named_parameters()}

    grad_dict = vector_to_gradients(param_vector, shapes)

    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(grad_dict[name])

    param_grad_sums = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    total_samples = 0

    for images, labels in dataloader:
        #images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        model.zero_grad()

        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_grad_sums[name] += param.grad.clone()
        total_samples += len(labels)

    avg_gradients = {name: grad_sum / total_samples for name, grad_sum in param_grad_sums.items()}

    return gradients_to_vector(avg_gradients)[0]