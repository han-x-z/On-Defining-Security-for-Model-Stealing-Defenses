import time
import argparse
import os.path as osp
import os
from datetime import datetime
import json

import torch
import admis.config as cfg
import admis.utils.datasets as datasets
import admis.utils.model as model_utils
import knockoff.utils.utils as knockoff_utils
import admis.utils.zoo as zoo

def main():
    parser = argparse.ArgumentParser(description='Train the Defender model')
    # Required arguments
    parser.add_argument('dataset', metavar='DS_NAME', type=str, help='Dataset name')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')

    # Optional arguments
    parser.add_argument('-o', '--out_path', metavar='PATH', type=str, help='Output path for model',
                        default=cfg.MODEL_DIR)
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr-step', type=int, default=50, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
    parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=None)

    # Defense
    parser.add_argument('--defense', metavar='DEF', type=str,
                        help='No Defense(ND)/ Prediction Poisoning(PP)/ Selective Misinformation (SM)/ Self Generator（SG）', default='ND')
    # Outlier Exposure
    parser.add_argument('--oe_lamb', type=float, default=0.0, metavar='LAMB',
                        help='Lambda for Outlier Exposure')
    parser.add_argument('-doe', '--dataset_oe', metavar='DS_OE_NAME', type=str, help='OE Dataset name',
                        default='Indoor67')
    

    args = parser.parse_args()
    params = vars(args)

    # torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Set up dataset
    dataset_name = params['dataset']
    valid_datasets = datasets.__dict__.keys()
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    if params['dataset'] == 'GTSRB':
        trainset = dataset(train=True, transform=train_transform)
        testset = dataset(train=False, transform=test_transform)
    else:
        trainset = dataset(train=True, transform=train_transform)
        testset = dataset(train=False, transform=test_transform)
    print("trainset:", len(trainset))
    print("testset:", len(testset))
    if params['dataset'] == 'GTSRB':
        num_classes = 43
    else:
        num_classes = len(trainset.classes)
    params['num_classes'] = num_classes
    # ----------- Set up model
    model_name = params['model_arch']
    pretrained = params['pretrained']
    params['out_path'] = out_path = '{}/{}/'.format(params['out_path'], params['defense'])
    out_path = params['out_path']

    model = zoo.get_net(model_name, modelfamily, pretrained, dataset_name, num_classes=num_classes)  # Defender's Model
    model = model.to(device)
    if not osp.exists(out_path):
        knockoff_utils.create_dir(out_path)

    trainset_oe = None
    testset_oe = None
    model_poison = None
    if params['defense'] == 'SM': # set up dataset for Outlier Exposure
        dataset_oe_name = params['dataset_oe']
        if dataset_oe_name not in valid_datasets:
            raise ValueError('OE Dataset not found. Valid arguments = {}'.format(valid_datasets))
        dataset_oe = datasets.__dict__[dataset_oe_name]
        modelfamily_oe = datasets.dataset_to_modelfamily[dataset_oe_name]
        train_oe_transform = datasets.modelfamily_to_transforms[modelfamily_oe]['train']
        test_oe_transform = datasets.modelfamily_to_transforms[modelfamily_oe]['test']
        trainset_oe = dataset_oe(train=True, transform=train_oe_transform)
        testset_oe = dataset_oe(train=False, transform=test_oe_transform)
        model_poison = zoo.get_net(model_name, modelfamily, pretrained,
                               num_classes=num_classes)  # Alt model for Selective Misinformation
        model_poison = model_poison.to(device)
    elif params['defense'] == 'SG': # set up dataset for Outlier Exposure
        dataset_oe_name = params['dataset_oe']
        if dataset_oe_name not in valid_datasets:
            raise ValueError('OE Dataset not found. Valid arguments = {}'.format(valid_datasets))
        dataset_oe = datasets.__dict__[dataset_oe_name]
        modelfamily_oe = datasets.dataset_to_modelfamily[dataset_oe_name]
        train_oe_transform = datasets.modelfamily_to_transforms[modelfamily_oe]['train']
        test_oe_transform = datasets.modelfamily_to_transforms[modelfamily_oe]['test']
        trainset_oe = dataset_oe(train=True, transform=train_oe_transform)
        testset_oe = dataset_oe(train=False, transform=test_oe_transform)
        model_poison = zoo.get_net(model_name, modelfamily, pretrained,
                               num_classes=num_classes)  # Alt model for Selective Misinformation
        model_poison = model_poison.to(device)
    elif params['defense'] == 'BCG': # set up dataset for Outlier Exposure
        dataset_oe_name = params['dataset_oe']
        if dataset_oe_name not in valid_datasets:
            raise ValueError('OE Dataset not found. Valid arguments = {}'.format(valid_datasets))
        dataset_oe = datasets.__dict__[dataset_oe_name]
        modelfamily_oe = datasets.dataset_to_modelfamily[dataset_oe_name]
        train_oe_transform = datasets.modelfamily_to_transforms[modelfamily_oe]['train']
        test_oe_transform = datasets.modelfamily_to_transforms[modelfamily_oe]['test']
        trainset_oe = dataset_oe(train=True, transform=train_oe_transform)
        testset_oe = dataset_oe(train=False, transform=test_oe_transform)
        model_poison = zoo.get_net(model_name, modelfamily, pretrained,
                               num_classes=num_classes)  # Alt model for Selective Misinformation
        model_poison = model_poison.to(device)
    elif params['defense'] == 'PP':
        model_pp = zoo.get_net(model_name, modelfamily, pretrained, num_classes=num_classes)  # Dummy model for predPoison
        torch.save(model_pp.state_dict(), out_path + '/model_pp.pt')

    # ----------- Train
    print('Starting Training..\n')
    model_utils.train_model(model, trainset=trainset, trainset_OE=trainset_oe, testset=testset, testset_OE=testset_oe,
                            model_poison=model_poison, device=device, **params)

    if params['defense'] == 'SM':
        torch.save(model_poison.state_dict(), out_path + '/model_poison.pt')
    elif params['defense'] == 'SG':
        torch.save(model_poison.state_dict(), out_path + '/model_poison.pt')
    elif params['defense'] == 'BCG':
        torch.save(model_poison.state_dict(), out_path + '/model_poison.pt')
    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_path, 'params.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    seconds = end-start
    hour = int(seconds/(60*60))
    min = int(seconds/60) % 60
    sec = int(seconds) % 60
    print('Runtime: {}:{}:{}'.format(hour, min, sec))
