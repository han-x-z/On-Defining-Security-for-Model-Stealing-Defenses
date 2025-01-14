import torch
import torch.nn as nn
import os.path as osp
import torchvision.models as models
import knockoff.models.mnist
import torch.nn.functional as F
from torchvision.models import vgg16_bn, resnet34
class Conv3(nn.Module):
    """A simple MNIST network

    Source: https://github.com/pytorch/examples/blob/master/mnist/main.py
    """
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.fc1 = nn.Linear(128*4*4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        batches = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = x.view([batches, -1])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*4*4, 256)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batches = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = x.view([batches, -1])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def conv3(num_classes, **kwargs):
    return Conv3(num_classes, **kwargs)

def binaryclassifier(num_classes, **kwargs):
    return BinaryClassifier(num_classes, **kwargs)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
model_dict={
    'lenet': knockoff.models.mnist.lenet,
    'conv3': conv3,
    'wrn': models.wide_resnet50_2,
    'squeeze': models.squeezenet1_1,
    'res18': models.resnet18,
    'vgg16': models.vgg16,
    'BinaryClassifier':binaryclassifier,
    'vgg16_bn': vgg16_bn,
    'resnet34':resnet34,
}

def get_net(modelname, modeltype, pretrained=None, dataset='', **kwargs):
    model_fn = model_dict[modelname]

    if modelname == 'res18':
        if pretrained:
            model = model_fn(pretrained=True)
        else:
            model = model_fn()
        if 'CIFAR' in dataset:
            model.conv1.stride = (1, 1)
            model.maxpool = Identity()
        in_feat = model.fc.in_features
        num_classes = kwargs['num_classes']
        model.fc = nn.Linear(in_feat, num_classes)
    elif modelname == 'conv3':
        model =model_fn(kwargs['num_classes'])
    elif modelname == 'binaryclassifier':
        model =model_fn(kwargs['num_classes'])
    elif modelname == 'lenet':
        model = model_fn(kwargs['num_classes'])
    elif modelname == 'vgg16':
        model = model_fn(kwargs['num_classes'])
    elif modelname == 'vgg16_bn':
        model = model_fn()
        num_classes = kwargs['num_classes']
        # in_feat = net.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    elif modelname == 'resnet34':
        if pretrained:
            model = model_fn(pretrained=True)
        else:
            model = model_fn()
        if 'CIFAR' in dataset:
            model.conv1.stride = (1, 1)
            model.maxpool = Identity()
        in_feat = model.fc.in_features
        num_classes = kwargs['num_classes']
        model.fc = nn.Linear(in_feat, num_classes)
    return model
