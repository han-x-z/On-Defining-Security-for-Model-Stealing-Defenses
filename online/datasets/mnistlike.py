import os.path as osp

from torchvision.datasets import MNIST as TVMNIST
from torchvision.datasets import EMNIST as TVEMNIST
from torchvision.datasets import FashionMNIST as TVFashionMNIST
from torchvision.datasets import KMNIST as TVKMNIST
from torchvision import datasets, transforms
import online.config as cfg


class MNIST(TVMNIST):
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        root = osp.join(cfg.DATASET_ROOT, 'mnist')
        super().__init__(root, train, transform, target_transform, download)


class KMNIST(TVKMNIST):
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        root = osp.join(cfg.DATASET_ROOT, 'kmnist')
        super().__init__(root, train, transform, target_transform, download)


class EMNIST(TVEMNIST):
    def __init__(self, **kwargs):
        root = osp.join(cfg.DATASET_ROOT, 'emnist')
        super().__init__(root, split='balanced', download=True, **kwargs)
        # Images are transposed by default. Fix this.
        self.data = self.data.permute(0, 2, 1)


class EMNISTLetters:
    def __init__(self, train=True, transform=None):
        root = osp.join(cfg.DATASET_ROOT, 'emnist')
        self.dataset = datasets.EMNIST(root, split='letters', train=train, download=True, transform=transform)
        self.data = self.dataset.data
    def __getitem__(self, index):
        # 返回给定索引处的数据和标签
        return self.dataset[index]


class FashionMNIST(TVFashionMNIST):
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        root = osp.join(cfg.DATASET_ROOT, 'mnist_fashion')
        super().__init__(root, train, transform, target_transform, download)
