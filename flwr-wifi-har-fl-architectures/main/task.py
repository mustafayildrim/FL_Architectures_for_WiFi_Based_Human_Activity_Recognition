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

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.4)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.4)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.4)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*31*11, 256),
            nn.ReLU(),
            # nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.fc(x)
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
abs_path = "C:\\Users\\katyc\\Documents\\GitHub\\FL_Architectures_for_WiFi_Based_Human_Activity_Recognition\\flwr-wifi-har-fl-architectures\\UT_HAR"
def load_UT_HAR(root_dir=abs_path):
    expected = ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]
    data = {}
    for folder in ["data", "label"]:
        files = glob.glob(os.path.join(root_dir, folder, "*.csv"))
        for path in files:
            key = os.path.basename(path).split(".")[0]
            if key not in expected:
                continue
            arr = np.load(path, allow_pickle=True)
            if folder == "data":
                arr = arr.reshape(len(arr), 1, 250, 90)
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                arr = torch.tensor(arr, dtype=torch.float32)
            else:
                arr = torch.tensor(arr, dtype=torch.long)
            data[key] = arr
    X_train, y_train = augment_dataset(data["X_train"], data["y_train"], factor=0)  # factor=0 means no extra copies
    data["X_train"], data["y_train"] = X_train, y_train
    return data

# Global cache for the har data
fds = None  # Cache FederatedDataset

def load_data(batch_size=64, root_dir=abs_path):
    har_data = load_UT_HAR(root_dir)
    X_train, y_train = har_data['X_train'], har_data['y_train']
    X_val, y_val = har_data['X_val'], har_data['y_val']
    X_test, y_test = har_data['X_test'], har_data['y_test']

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def load_partitioned_data(partition_id: int, num_partitions: int, batch_size: int, root_dir=abs_path):
    har_data = load_UT_HAR(root_dir)

    X = har_data["X_train"]
    y = har_data["y_train"]
    
    # Shuffle the data before partitioning (IID federated learning)
    n = len(X)
    perm = torch.randperm(n)
    X = X[perm]
    y = y[perm]

    part_size = n // num_partitions

    start = partition_id * part_size
    end = (partition_id + 1) * part_size if partition_id < num_partitions - 1 else n
    X_part = X[start:end]
    y_part = y[start:end]

    split = int(0.8 * len(X_part))
    X_train, X_test = X_part[:split], X_part[split:]
    y_train, y_test = y_part[:split], y_part[split:]

    trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    testloader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return trainloader, testloader

def load_centralized_dataset(batch_size=128, root_dir=abs_path):
    har_data = load_UT_HAR(root_dir)
    dataset = TensorDataset(har_data['X_test'], har_data['y_test'])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def train(net, trainloader, device, epochs=5, lr=1e-3):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    net.train()
    print("Starting training...")

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
