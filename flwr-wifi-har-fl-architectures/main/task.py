"""pytorchexample: A Flower / PyTorch app."""

import os

import torch
import glob
import numpy as np
from datasets import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, densenet121
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader, TensorDataset, Subset

# Define a custom CNN model
class CNNModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 31 * 11, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

# Define a ResNet50 model
class ResNet50Model(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.base = resnet50(weights=None)
        self.base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x):
        return self.base(x)
    
# Define a DenseNet121 model
class DenseNetModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.base = densenet121(weights=None)
        self.base.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base.classifier = nn.Linear(self.base.classifier.in_features, num_classes)

    def forward(self, x):
        return self.base(x)

# Define a simple data augmentation pipeline for the WiFi HAR data
class DataAugmentation:
    def __init__(self, noise=0.01, max_shift=5, scale_range=(0.9, 1.1)):
        self.noise = noise
        self.max_shift = max_shift
        self.scale_range = scale_range

    def __call__(self, x):
        x = x + self.noise * torch.randn_like(x)
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        x = torch.roll(x, shifts=shift, dims=2)
        x = x * torch.empty(1).uniform_(*self.scale_range).item()
        return x

def augment_dataset(X, y, factor=2):
    aug = DataAugmentation()
    X_list, y_list = [X], [y]
    X_list += [torch.stack([aug(x) for x in X]) for _ in range(factor)]
    y_list += [y for _ in range(factor)]
    return torch.cat(X_list, dim=0), torch.cat(y_list, dim=0)

# Load the UT-HAR dataset, which is already partitioned into train/val/test sets
def load_UT_HAR(root_dir):
    data_files = glob.glob(root_dir + '/data/*.csv')
    label_files = glob.glob(root_dir + '/label/*.csv')
    expected = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
    wifi = {}

    for path in data_files:
        key = path.replace('\\', '/').split('/')[-1].split('.')[0]
        if key not in expected:
            continue
        data = np.load(path, allow_pickle=True)
        data = data.reshape(len(data), 1, 250, 90)
        data = (data - data.min()) / (data.max() - data.min())
        wifi[key] = torch.tensor(data, dtype=torch.float32)

    for path in label_files:
        key = path.replace('\\', '/').split('/')[-1].split('.')[0]
        if key not in expected:
            continue
        labels = np.load(path, allow_pickle=True)
        wifi[key] = torch.tensor(labels, dtype=torch.int64)
        
    # Exapnd the training set with data augmentation
    # wifi['X_train'], wifi['y_train'] = augment_dataset(wifi['X_train'], wifi['y_train'], factor=3)

    return wifi

# Global cache for the har data
har_data = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'UT_HAR')

def load_data():
    wifi = load_UT_HAR(DATA_DIR)

    X_train, y_train = wifi['X_train'], wifi['y_train']
    X_test, y_test = wifi['X_test'], wifi['y_test']

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

def load_partitioned_data(partition_id: int, num_partitions: int, batch_size: int):
    global har_data, partitioner

    if har_data is None:
        har_data = load_UT_HAR(DATA_DIR)

        hf_dataset = Dataset.from_dict({
            "image": har_data["X_train"].numpy(),
            "label": har_data["y_train"].numpy()
        })

        partitioner = IidPartitioner(num_partitions=num_partitions)
        partitioner.dataset = hf_dataset

    # Load client partition
    client_data = partitioner.load_partition(partition_id)

    # Convert to torch
    X = torch.tensor(client_data["image"], dtype=torch.float32)
    y = torch.tensor(client_data["label"], dtype=torch.int64).squeeze()

    train_dataset = TensorDataset(X, y)
    test_dataset = TensorDataset(
        har_data["X_test"], har_data["y_test"].squeeze()
    )

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size)

    return trainloader, testloader


def load_centralized_dataset(batch_size=128):
    global har_data

    if har_data is None:
        har_data = load_UT_HAR(DATA_DIR)

    test_dataset = TensorDataset(
        har_data["X_test"], har_data["y_test"].squeeze()
    )

    return DataLoader(test_dataset, batch_size=batch_size)


def train(net, trainloader, device, epochs=3, lr=1e-3):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    net.train()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        correct = 0
        total = 0

        for signals, labels in trainloader:
            signals = signals.to(device)
            labels = labels.squeeze().long().to(device)

            optimizer.zero_grad()
            outputs = net(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = total_loss / n_batches
        epoch_acc = correct / total

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    return epoch_loss, epoch_acc


def test(net, testloader, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    correct = 0
    total_samples = 0
    loss = 0.0

    net.eval()

    with torch.no_grad():
        for signals, labels in testloader:
            signals = signals.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = net(signals)

            batch_size = labels.size(0)
            loss += criterion(outputs, labels).item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()

    accuracy = correct / total_samples
    loss = loss / total_samples

    return loss, accuracy
