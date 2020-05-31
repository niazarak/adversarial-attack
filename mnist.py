import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torchvision.datasets import MNIST


class MnistClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 5, 5, 1)
        self.conv2 = torch.nn.Conv2d(5, 10, 5, 1)

        self.linear_1 = torch.nn.Linear(4 * 4 * 10, 128)
        self.linear_2 = torch.nn.Linear(128, 10)

    def forward(self, *input):
        out = input[0].view(-1, 1, 28, 28)
        out = nn.functional.relu(self.conv1(out))
        out = nn.functional.max_pool2d(out, [2, 2])

        out = nn.functional.relu(self.conv2(out))
        out = nn.functional.max_pool2d(out, [2, 2])

        out = out.view(-1, 4 * 4 * 10)
        out = self.linear_1(out)
        out = torch.sigmoid(out)
        out = self.linear_2(out)
        out = torch.sigmoid(out)
        return out


class MnistDataset(Dataset):
    def __init__(self, mnist: MNIST):
        self.mnist = mnist

    def __getitem__(self, index):
        return {
            'data': self.mnist[index][0],
            'label': self.mnist[index][1]
        }

    def __len__(self):
        return len(self.mnist)


def transform(i):
    data = torch.tensor(i.getdata())
    data = data / 255.0
    return data


def target_transform(t):
    return torch.tensor(t)


def load_data(train: bool) -> MNIST:
    return MNIST('.', transform=transform, target_transform=target_transform, train=train, download=True)
