import argparse
import os
import sys
sys.path.append("/workspace/D-DAE")
from torch.utils.data import Subset
from online.victim import *
from online.victim.bb_BCG import BCG
from online.victim.bb_SG import SG

from online import datasets

import os.path as osp
from offline.recovery import Restorer
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from tqdm import tqdm
BBOX_CHOICES = ['none', 'topk', 'rounding',
                'reverse_sigmoid', 'reverse_sigmoid_wb',
                'rand_noise', 'rand_noise_wb',
                'mad', 'mad_wb', 'edm', 'am', 'bcg','sg']
def parse_defense_kwargs(kwargs_str):
    kwargs = dict()
    for entry in kwargs_str.split(','):
        if len(entry) < 1:
            continue
        key, value = entry.split(':')
        assert key not in kwargs, 'Argument ({}:{}) conflicts with ({}:{})'.format(key, value, key, kwargs[key])
        try:
            # Cast into int if possible
            value = int(value)
        except ValueError:
            try:
                # Try with float
                value = float(value)
            except ValueError:
                # Give up
                pass
        kwargs[key] = value
    return kwargs

class Restorer(nn.Module):
    def __init__(self, num_classes=43):
        super(Restorer, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Tanh(),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        output = self.main(input)
        return output

    def predict(self, input):
        output = self.main(input)
        output /= output.sum(dim=1)[:, None]
        return output

def main():
    parser = argparse.ArgumentParser(description='Construct transfer set')
    parser.add_argument('policy', metavar='PI', type=str, help='Policy to use while training',
                        choices=['random', 'adaptive'])
    parser.add_argument('victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('defense', metavar='TYPE', type=str, help='Type of defense to use',
                        choices=BBOX_CHOICES, default='none')
    parser.add_argument('defense_args', metavar='STR', type=str, help='Blackbox arguments in format "k1:v1,k2:v2,..."')
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store transfer set', required=True)
    parser.add_argument('--budget', metavar='N', type=int, help='# images',
                        default=None)
    parser.add_argument('--nqueries', metavar='N', type=int, help='# queries to blackbox using budget images',
                        default=None)
    parser.add_argument('--qpi', metavar='N', type=int, help='# queries per image',
                        default=1)
    parser.add_argument('--queryset', metavar='TYPE', type=str, help='Adversary\'s dataset (P_A(X))', required=True)
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=1)
    # ----------- Other params
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--train_transform', action='store_true', help='Perform data augmentation', default=False)
    args = parser.parse_args()
    params = vars(args)

    out_path = params['out_dir']

    def create_dir(dir_path):
        if not osp.exists(dir_path):
            print('Path {} does not exist. Creating it...'.format(dir_path))
            os.makedirs(dir_path)
    create_dir(out_path)

    torch.manual_seed(42)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Set up queryset
    queryset_name = params['queryset']
    valid_datasets = datasets.__dict__.keys()
    if queryset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    modelfamily = datasets.dataset_to_modelfamily[queryset_name]
    transform_type = 'train' if params['train_transform'] else 'test'
    if params['train_transform']:
        print('=> Using data augmentation while querying')
    transform = datasets.modelfamily_to_transforms[modelfamily][transform_type]
    queryset = datasets.__dict__[queryset_name](train=True, transform=transform)
    import random
    N = len(queryset)
    subset_indices = random.sample(range(N), 1000)
    subset = Subset(queryset, subset_indices)
    test_loader = DataLoader(subset, batch_size=1 ,shuffle=False)
    data_loader_size = len(test_loader)
    print("DataLoader size:", data_loader_size)
    if params['budget'] is None:
        params['budget'] = len(queryset)
    # ----------- Initialize blackbox
    blackbox_dir = params['victim_model_dir']
    defense_type = params['defense']
    if defense_type == 'rand_noise':
        BB = RandomNoise
    elif defense_type == 'mad':
        BB = MAD
    elif defense_type == 'reverse_sigmoid':
        BB = ReverseSigmoid
    elif defense_type in ['none', 'topk', 'rounding']:
        BB = Blackbox
    elif defense_type == 'edm':
        BB = EDM_device
    elif defense_type == 'am':
        BB = AM
    elif defense_type == 'bcg':
        BB = BCG
    elif defense_type == 'sg':
        BB = SG
    else:
        raise ValueError('Unrecognized blackbox type')
    defense_kwargs = parse_defense_kwargs(params['defense_args'])
    defense_kwargs['log_prefix'] = 'transfer'
    print('=> Initializing BBox with defense {} and arguments: {}'.format(defense_type, defense_kwargs))
    #blackbox1 = Blackbox.from_modeldir('models/victim/CIFAR-10-vgg16-bn-train-nodefense', device=torch.device("cuda"))
    #blackbox1 = Blackbox.from_modeldir('models/victim/GTSRB-vgg16_bn-train-nodefense', device=torch.device("cuda"))
    blackbox1 = Blackbox.from_modeldir('models/victim/ImageNette-resnet34-train-nodefense', device=torch.device("cuda"))
    #blackbox1 = Blackbox.from_modeldir('models/victim/MNIST-lenet-train-nodefense', device=torch.device("cuda"))
    blackbox2 = BB.from_modeldir(blackbox_dir, device=torch.device("cuda"), **defense_kwargs)
    for k, v in defense_kwargs.items():
        params[k] = v

    # ----------- Initialize adversary
    batch_size = params['batch_size']
    nworkers = params['nworkers']
    outputs_t1 = []
    outputs_t2 = []
    batch_counter = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = blackbox1(inputs).cuda()
                #print(outputs)
            outputs_t1.append(outputs)
        #print(outputs_t1)
        y_t1 = torch.stack(outputs_t1)  # Mean over queries
        y_t1 = y_t1.view(-1, y_t1.size(2))
        #print(y_t1)
    batch_counter = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = blackbox2(inputs).cuda()
            outputs_t2.append(outputs)

        y_t2 = torch.stack(outputs_t2)  # Mean over queries
        y_t2 = y_t2.view(-1, y_t2.size(2))
        #print("Size of y_t2:", y_t2.size())
    # 提取硬标签
    restorer = Restorer(num_classes=10).to(device)  # TODO
    #generator_path = 'generator/cifar10/BCG'#TODO
    #generator_path = 'generator/mnist/BCG'
    #generator_path = 'generator/gtsrb/BCG'
    generator_path = 'generator/imagenette/BCG'
    restorer.load_state_dict(torch.load(generator_path + '/Restorer_epoch_8.pth'))
    #restorer.load_state_dict(torch.load(generator_path + '/Restorer.pth'))
    y_t3 = restorer(y_t2.detach()) #TODO
    _, predicted_labels_t1 = y_t1.max(dim=1)
    _, predicted_labels_t2 = y_t2.max(dim=1)
    _, predicted_labels_t3 = y_t3.max(dim=1)
    print(" y_t1:", y_t1)
    print(" y_t2:", y_t2)
    print(" y_t3:", y_t3)
    print("Predicted labels for y_t1:", predicted_labels_t1)
    print("Predicted labels for y_t2:", predicted_labels_t2)
    print("Predicted labels for y_t3:", predicted_labels_t3)

    # 比较硬标签
    matches1 = predicted_labels_t1 == predicted_labels_t2
    matches2 = predicted_labels_t1 == predicted_labels_t3
    # 计算相同硬标签的数量
    num_matches1 = matches1.sum().item()
    num_matches2 = matches2.sum().item()
    blackbox2.defense_fn.get_stats()
    print(f"相同硬标签的数量1: {num_matches1}")
    print(f"相同硬标签的数量2: {num_matches2}")
if __name__ == '__main__':
    main()
