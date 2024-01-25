import argparse
import os
import os.path as osp
import json
from datetime import datetime
import sys
sys.path.append("/workspace/D-DAE")
import torch
from torch.utils.data import DataLoader
from online import datasets
import online.utils.utils as knockoff_utils
import online.config as cfg
from online.victim import *
from online.adversary.transfer import BBOX_CHOICES, parse_defense_kwargs
from online.victim.bb_BCG import BCG
def test_defense(victim_model, test_loader, device, epoch=0., silent=False):
    victim_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_victim = victim_model(inputs)
            _, predicted_victim = outputs_victim.max(1)
            total += targets.size(0)
            correct += predicted_victim.eq(targets).sum().item()

    accuracy = 100. * correct / total

    if not silent:
        print('[Test]  Epoch: {}\tAccuracy: {:.1f}% ({}/{})'.format(
            epoch, accuracy, correct, total))

    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Test defense effectiveness')

    parser.add_argument(
        'victim_model_dir',
        metavar='PATH',
        type=str,
        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"'
    )

    parser.add_argument('defense',
                        metavar='TYPE',
                        type=str,
                        help='Type of defense to use',
                        choices=BBOX_CHOICES)
    parser.add_argument('defense_args',
                        metavar='STR',
                        type=str,
                        help='Blackbox arguments in format "k1:v1,k2:v2,..."')
    parser.add_argument('--batch_size',
                        metavar='TYPE',
                        type=int,
                        help='Batch size of queries',
                        default=128)
    parser.add_argument('-d',
                        '--device_id',
                        metavar='D',
                        type=int,
                        help='Device id',
                        default=0)
    parser.add_argument('-w',
                        '--nworkers',
                        metavar='N',
                        type=int,
                        help='# Worker threads to load data',
                        default=10)
    args = parser.parse_args()
    params = vars(args)

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Initialize blackbox
    blackbox_dir = params['victim_model_dir']
    defense_type = params['defense']

    if defense_type == 'rand_noise':
        BB = RandomNoise
    elif defense_type == 'rand_noise_wb':
        BB = RandomNoise_WB
    elif defense_type == 'mad':
        BB = MAD
    elif defense_type == 'edm':
        BB = EDM_device
    elif defense_type == 'am':
        BB = AM
    elif defense_type == 'mad_wb':
        BB = MAD_WB
    elif defense_type == 'reverse_sigmoid':
        BB = ReverseSigmoid
    elif defense_type == 'reverse_sigmoid_wb':
        BB = ReverseSigmoid_WB
    elif defense_type in ['none', 'topk', 'rounding']:
        BB = Blackbox
    elif defense_type == 'bcg':
        BB = BCG
    else:
        raise ValueError('Unrecognized blackbox type')
    defense_kwargs = parse_defense_kwargs(params['defense_args'])
    defense_kwargs['log_prefix'] = 'test'
    print('=> Initializing BBox with defense {} and arguments: {}'.format(
        defense_type, defense_kwargs))
    blackbox = BB.from_modeldir(blackbox_dir, torch.device('cuda'),
                                **defense_kwargs)

    for k, v in defense_kwargs.items():
        params[k] = v

    # ----------- Set up queryset
    with open(osp.join(blackbox_dir, 'params.json'), 'r') as rf:
        bbox_params = json.load(rf)
    testset_name = bbox_params['dataset']
    valid_datasets = datasets.__dict__.keys()
    if testset_name not in valid_datasets:
        raise ValueError(
            'Dataset not found. Valid arguments = {}'.format(valid_datasets))

    modelfamily = datasets.dataset_to_modelfamily[testset_name]
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    testset = datasets.__dict__[testset_name](train=False, transform=transform)
    print('=> Evaluating on {} ({} samples)'.format(testset_name,
                                                    len(testset)))

    # ----------- Evaluate
    batch_size = params['batch_size']
    nworkers = params['nworkers']
    epoch = bbox_params['epochs']

    # ----------- Evaluate
    testloader = DataLoader(testset,
                            num_workers=nworkers,
                            shuffle=False,
                            batch_size=batch_size)
    accuracy = test_defense(blackbox, testloader, device, epoch=epoch)

    # ... (Logging and saving results)

if __name__ == '__main__':
    main()
