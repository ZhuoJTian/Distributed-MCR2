from torch.utils.data import Dataset
import numpy as np
import torchvision
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class MyAffineTransform:
    """Transform by one of the ways."""
    '''Choice:[angle, scale]'''

    def __init__(self, choice):
        self.choice = choice

    def __call__(self, x):
        choice = self.choice
        x = F.affine(x, angle=choice[0], scale=choice[1], translate=[0, 0], shear=0)
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
            x = self.transform(x.numpy().astype(np.uint8)) # 
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y


def create_datasets(data_path, dataset_name, num_clients, choices, local_dist, local_dist_test):
    """Split the whole dataset for distributing to clients."""
    training_dataset = torchvision.datasets.MNIST(root=data_path, train=True, transform=None, download=True)
    
    test_dataset = torchvision.datasets.MNIST(root=data_path, train=False, transform=None, download=True)
    # unsqueeze channel dimension for grayscale image datasets
    
    if "ndarray" not in str(type(training_dataset.data)):
        training_dataset.data = np.asarray(training_dataset.data)
    if "list" not in str(type(training_dataset.targets)):
        training_dataset.targets = training_dataset.targets.tolist()

    y_train = [[item] for item in training_dataset.targets]
    y_test = [[item] for item in test_dataset.targets]
	
    sample_index_train = [[] for _ in range(10)]
    for category in range(10):
        for index in range(len(y_train)):
            if y_train[index][0] == category:
                sample_index_train[category].append(index)

    sample_index_test = [[] for _ in range(10)]
    for category in range(10):
        for index in range(len(y_test)):
            if y_test[index][0] == category:
                sample_index_test[category].append(index)

    # split training dataset according to label distribution

    # get the indices of training dataset for each client
    idx_clients_train = [[] for i in range(num_clients)]
    remained = [0 for _ in range(10)]
    for i in range(num_clients):
        index = []
        for cls in range(10):
            if local_dist[i][cls] > 0:
                for num in range(int(remained[cls]), int(remained[cls] + int(local_dist[i][cls]))):
                    index.append(int(sample_index_train[cls][num]))
                remained[cls] += int(local_dist[i][cls])
        idx_clients_train[i] = index

    # get the indices of testing dataset for each client
    idx_clients_test = [[] for i in range(num_clients)]
    remained = [0 for _ in range(10)]
    for i in range(num_clients):
        index = []
        for cls in range(10):
            if local_dist_test[i][cls] > 0:
                for num in range(int(remained[cls]), int(remained[cls] + local_dist_test[i][cls])):
                    index.append(int(sample_index_test[cls][num]))
                # remained[cls] += local_dist_test[i][cls]
        idx_clients_test[i] = index

    # get the indices of all training samples in each client
    idx_clients_trainall = [[] for i in range(num_clients)]
    remained = [0 for _ in range(10)]
    for i in range(num_clients):
        index = []
        for cls in range(10):
            if local_dist[i][cls] > 0:
                for num in range(int(remained[cls]), int(remained[cls] + int(local_dist[i][cls]))):
                    index.append(int(sample_index_train[cls][num]))
                remained[cls] += int(local_dist[i][cls])
        idx_clients_trainall[i] = index

    split_datasets_train = []
    split_datasets_train_all = []
    split_datasets_test = []
    local_datasets_train = []
    local_datasets_train_all = []
    local_datasets_test = []

    for client_id in range(num_clients):
        transform = transforms.Compose(
            [transforms.ToPILImage(), 
            transforms.Resize(32),
            transforms.ToTensor(),
            MyAffineTransform(choices[client_id]),
            transforms.Normalize(0.5, 0.5)],
            )

        split_datasets_train.append(((torch.Tensor(training_dataset.data)[torch.tensor(idx_clients_train[client_id])]),
                            (torch.Tensor(training_dataset.targets)[torch.tensor(idx_clients_train[client_id])]).long()))

        split_datasets_train_all.append(((torch.Tensor(training_dataset.data)[torch.tensor(idx_clients_trainall[client_id])]),
                                     (torch.Tensor(training_dataset.targets)[
                                         torch.tensor(idx_clients_trainall[client_id])]).long()))

        split_datasets_test.append(((torch.Tensor(test_dataset.data)[torch.tensor(idx_clients_test[client_id])]),
                            (torch.Tensor(test_dataset.targets)[torch.tensor(idx_clients_test[client_id])]).long()))

        local_datasets_train.append(CustomTensorDataset(split_datasets_train[client_id], transform=transform))
        # local_datasets_train_all.append(CustomTensorDataset(split_datasets_train_all[client_id], transform=transform))
        local_datasets_test.append(CustomTensorDataset(split_datasets_test[client_id], transform=transform))

    return local_datasets_train, local_datasets_train_all, local_datasets_test
