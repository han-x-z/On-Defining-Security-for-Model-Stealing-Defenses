from torchvision import transforms

from knockoff.datasets.caltech256 import Caltech256
from knockoff.datasets.cifarlike import CIFAR10, CIFAR100, SVHN
from knockoff.datasets.cubs200 import CUBS200
from knockoff.datasets.diabetic5 import Diabetic5
from knockoff.datasets.imagenet1k import ImageNet1k
from knockoff.datasets.indoor67 import Indoor67
from knockoff.datasets.mnistlike import MNIST, KMNIST, EMNIST, EMNISTLetters, FashionMNIST
from knockoff.datasets.tinyimagenet200 import TinyImageNet200
from admis.utils.tinyimages_80mn_loader import TinyImages
from admis.utils.dtd_loader import DTD
from admis.utils.flowers_loader import Flowers17, Flowers102
from admis.utils.gtsrb import GTSRB
from admis.utils.ImageNette import ImageNette
from admis.utils.Imagewoof import ImageWoof

# Create a mapping of dataset -> dataset_type
# This is helpful to determine which (a) family of model needs to be loaded e.g., imagenet and
# (b) input transform to apply
dataset_to_modelfamily = {
    # MNIST
    'MNIST': 'mnist',
    'KMNIST': 'mnist',
    'EMNIST': 'mnist',
    'EMNISTLetters': 'mnist',
    'FashionMNIST': 'mnist',

    # Cifar
    'CIFAR10': 'cifar',
    'CIFAR100': 'cifar',
    'SVHN': 'cifar',
    'GTSRB': 'cifar',
    'LISA': 'cifar',
    'TinyImageNet200': 'cifar',
    'TinyImagesSubset': 'cifar',

    # Imagenet
    'CUBS200': 'imagenet',
    'Caltech256': 'imagenet',
    'Indoor67': 'imagenet',
    'Diabetic5': 'imagenet',
    'ImageNet1k': 'imagenet',
    'ImageFolder': 'imagenet',
    'ImageNette': 'imagenet',
    'miniImageNet': 'imagenet',
    'ImageWoof': 'imagenet',

    
    'Flowers17': 'flowers',
    'Flowers102': 'flowers',
}

modelfamily_to_mean_std = {
    'mnist': {
        'mean': (0.1307,),
        'std': (0.3081,),
    },
    'cifar': {
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010),
    },
    'imagenet': {
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
    }
}

# Transforms
modelfamily_to_transforms = {
    'mnist': {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
    },

    'cifar': {
        'train': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 #std=(0.2023, 0.1994, 0.2010)),
                                 std=(0.2470, 0.2435, 0.2616)),
        ]),
        'test': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 #std=(0.2023, 0.1994, 0.2010)),
                                 std=(0.2470, 0.2435, 0.2616)),
        ])
    },
     'flowers': {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ])
    },
    'imagenet': {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    }
}
