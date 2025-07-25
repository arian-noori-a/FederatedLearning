# loader.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np
from collections import defaultdict

def load_cifar10(data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = datasets.CIFAR10(data_dir, train=True, download=False, transform=transform)
    test_set = datasets.CIFAR10(data_dir, train=False, download=False, transform=transform)
    return train_set, test_set

def partition_dataset(train_dataset, num_clients=10, alpha=0.5):
    """Partition dataset in a non-IID manner using Dirichlet distribution."""
    label_indices = defaultdict(list)
    for idx, (_, label) in enumerate(train_dataset):
        label_indices[label].append(idx)

    client_indices = [[] for _ in range(num_clients)]
    for label in range(10):
        idxs = np.array(label_indices[label])
        np.random.shuffle(idxs)
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        splits = np.split(idxs, proportions)
        for i in range(num_clients):
            client_indices[i].extend(splits[i])

    return [Subset(train_dataset, indices) for indices in client_indices]
