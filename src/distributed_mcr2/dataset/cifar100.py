from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image


class MyAffineTransform:
    """Transform by one of the ways."""
    '''Choice:[angle, scale]'''

    def __init__(self, choice):
        self.choice = choice

    def __call__(self, x):
        angle, scale = self.choice
        x = F.affine(x, angle=angle, scale=scale, translate=[0, 0], shear=0)
        return x


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]

        if self.transform:
            x = Image.fromarray(x.numpy().astype(np.uint8))
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def create_datasets(data_path, dataset_name, num_clients, choices):

    # -----------------------------
    # Load CIFAR100
    # -----------------------------
    training_dataset = torchvision.datasets.CIFAR100(
        root=data_path,
        train=True,
        download=True
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root=data_path,
        train=False,
        download=True
    )

    NUM_CLASSES = 100

    train_data = torch.tensor(training_dataset.data)
    train_targets = torch.tensor(training_dataset.targets)

    test_data = torch.tensor(test_dataset.data)
    test_targets = torch.tensor(test_dataset.targets)

    # -----------------------------
    # build class index
    # -----------------------------
    train_class_idx = [[] for _ in range(NUM_CLASSES)]
    test_class_idx = [[] for _ in range(NUM_CLASSES)]

    for idx, label in enumerate(training_dataset.targets):
        train_class_idx[label].append(idx)

    for idx, label in enumerate(test_dataset.targets):
        test_class_idx[label].append(idx)

    # shuffle each class
    for c in range(NUM_CLASSES):
        np.random.shuffle(train_class_idx[c])
        np.random.shuffle(test_class_idx[c])

    # -----------------------------
    # strict IID split
    # -----------------------------
    train_per_client = 500 // num_clients
    test_per_client = 100 // num_clients

    idx_clients_train = [[] for _ in range(num_clients)]
    idx_clients_test = [[] for _ in range(num_clients)]

    for cls in range(NUM_CLASSES):

        for client in range(num_clients):

            start_train = client * train_per_client
            end_train = (client + 1) * train_per_client

            start_test = client * test_per_client
            end_test = (client + 1) * test_per_client

            idx_clients_train[client].extend(
                train_class_idx[cls][start_train:end_train]
            )

            idx_clients_test[client].extend(
                test_class_idx[cls][start_test:end_test]
            )

    # -----------------------------
    # build dataset
    # -----------------------------
    local_datasets_train = []
    local_datasets_test = []
    local_datasets_train_all = []

    for client_id in range(num_clients):

        transform = transforms.Compose([
            MyAffineTransform(choices[client_id]),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761)
            )
        ])

        train_idx = idx_clients_train[client_id]
        test_idx = idx_clients_test[client_id]

        train_dataset = (
            train_data[train_idx],
            train_targets[train_idx].long()
        )

        test_dataset = (
            test_data[test_idx],
            test_targets[test_idx].long()
        )

        local_datasets_train.append(
            CustomTensorDataset(train_dataset, transform)
        )

        local_datasets_test.append(
            CustomTensorDataset(test_dataset, transform)
        )

        local_datasets_train_all.append(train_dataset)

    return local_datasets_train, local_datasets_train_all, local_datasets_test