import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import uniform
import torch.nn as nn
import random
from torchvision import models
import os.path as osp
import json
import numpy as np
import torch
import torch.nn.functional as F
import online.models.zoo as zoo
from online import datasets
from online.victim import Blackbox

class BCG(Blackbox):
    def __init__(self, model, model_def, defense_levels=0.0, device=None, num_classes=10, rand_fhat=10, use_adaptive=True, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        self.device = torch.device('cuda') if device is None else device
        self.model = model.to(device)
        self.model.eval()
        self.__call_count = 0
        model_def = model_def.to(device)
        self.num_classes = num_classes
        self.defense_fn = bcGenerator(delta_list=0.0)


    @classmethod
    def from_modeldir(cls, model_dir, device=None, rand_fhat=False, use_adaptive=True, output_type='probs', **kwargs):
        device = torch.device('cuda') if device is None else device
        param_path = osp.join(model_dir, 'params.json')
        with open(param_path) as jf:
            params = json.load(jf)
        model_arch = params["model_arch"]
        num_classes = params["num_classes"]
        if 'queryset' in params:
            dataset_name = params['queryset']
        elif 'testdataset' in params:
            dataset_name = params['testdataset']
        elif 'dataset' in params:
            dataset_name = params['dataset']
        modelfamily = datasets.dataset_to_modelfamily[dataset_name]

        model = zoo.get_net(model_arch, modelfamily, num_classes=num_classes)
        model_def = zoo.get_net(model_arch, modelfamily, num_classes=num_classes)

        model = model.to(device)

        # Load Weights
        checkpoint_path = osp.join(model_dir, 'model_best.pth.tar')
        if not osp.exists(checkpoint_path):
            checkpoint_path = osp.join(model_dir, "checkpoint.pth.tar")
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint["epoch"]
        best_test_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["state_dict"])
        print(
            "=> loaded checkpoint (epoch {}, acc={:.2f})".format(epoch, best_test_acc)
        )
        model_def_path = model_dir + '/model_poison.pt'
        if not rand_fhat:
            print("loading BCG model")
            model_def.load_state_dict(torch.load(model_def_path))
        model_def = model_def.to(device)
        print(
            "=> loaded checkpoint for BCG model"
        )

        blackbox = cls(model=model, model_def=model_def,
                       output_type=output_type,
                       dataset_name=dataset_name,
                       defense_levels=0.0, device=device, num_classes=num_classes,
                       rand_fhat=rand_fhat,
                       use_adaptive=use_adaptive)
        return blackbox

    def __call__(self, x, T=1):
        with torch.no_grad():
            x = x.to(self.device)
            y = self.model(x)
            self.__call_count += x.shape[0]
            y = F.softmax(y/T, dim=1)
        y_mod = self.defense_fn(x, y)
        y_mod = list(y_mod.values())[0]
        return y_mod
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Generator, self).__init__()
        self.label_embed = nn.Embedding(num_classes, latent_dim)
        self.generator = nn.Sequential(
            nn.Linear(latent_dim * 2 , 128),  # 将噪音维度加入生成器输入维度
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, labels, noise):
        gen_input = self.label_embed(labels)
        gen_input_with_noise = torch.cat((gen_input, noise), -1)  # 将噪音和标签嵌入向量连接起来
        class_probs = self.generator(gen_input_with_noise)
        return class_probs

class MultiClassifier(nn.Module):
    def __init__(self):
        super(MultiClassifier,self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3,padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv5 = nn.Conv2d(128,128, 3,padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3,padding=1)
        self.conv7 = nn.Conv2d(128, 128, 1,padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv8 = nn.Conv2d(128, 256, 3,padding=1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.conv11 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        self.fc14 = nn.Linear(512*4*4,1024)
        self.drop1 = nn.Dropout2d()
        self.fc15 = nn.Linear(1024,1024)
        self.drop2 = nn.Dropout2d()
        self.fc16 = nn.Linear(1024,3)


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)


        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        # print(" x shape ",x.size())
        x = x.view(-1,512*4*4)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)

        return x




class bcGenerator:
    def __init__(
        self, delta_list, num_classes=10, #TODO
    ):
        self.delta_list = delta_list
        self.input_count = 0
        self.ood_count = 0
        self.num_classes = num_classes
        self.latent_dim = 10  # 定义 latent_dim 的值为 100
        self.mis_correct_count = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else False)
        self.generator = Generator(self.latent_dim, num_classes).to(self.device)
        self.MultiClassifier = MultiClassifier().to(self.device)
        self.model = models.vgg16(pretrained=True).to(self.device)
        self.model1 = models.resnet34(pretrained=True).to(self.device)

    def __call__(self, x, y):
        probs = y  # batch x 10
        prob = {}
        probs_max, probs_max_index = torch.max(probs, dim=1)  # batch
        known_label = probs_max_index
        batch = probs_max.size(0)
        self.input_count += batch
        #self.generator.load_state_dict(torch.load('./../adaptive_misinformation/CGAN-Pytorch/res/gtsrb/1/generator20.pth'))
        #self.generator.load_state_dict(torch.load('./../adaptive_misinformation/CGAN-Pytorch/res/cifar/1/generator70.pth'))
        self.generator.load_state_dict(torch.load('./../adaptive_misinformation/sampler/res/mnist/1/generator130.pth'))
        #self.generator.load_state_dict(torch.load('./../adaptive_misinformation/CGAN-Pytorch/res/image/1/generator140.pth'))
        self.generator.eval()
        # TODO
        #self.model.classifier[6] = torch.nn.Linear(4096, 3).to(self.device)
        #self.model.load_state_dict(torch.load('./../adaptive_misinformation/admis/utils/gtsrb/attackgtsrb1.pth'))
        #self.model.load_state_dict(torch.load('./../adaptive_misinformation/admis/utils/cifar/attackcifar4.pth'))
        # 修改全连接层以进行3分类
        #num_ftrs = self.model1.fc.in_features
        #self.model1.fc = nn.Linear(num_ftrs, 3).to(self.device)  # 3分类
        #self.model1.load_state_dict(torch.load('./../adaptive_misinformation/admis/utils/image/attackimage1.pth'))
        self.MultiClassifier.load_state_dict(torch.load('./../adaptive_misinformation/detector/mnist/attackmnist3.pth'))
        self.MultiClassifier.eval()
        # TODO
        gen_probs = {}
        gen_probs[0] = [None] * batch
        with torch.no_grad():
            gen_probs_list = []
            #outputs = self.model(x).to(self.device)
            outputs = self.MultiClassifier(x)
            #outputs = self.model1(x).to(self.device)
            # TODO
            _, predicted = torch.max(outputs, 1)
            #            print(predicted)
            for i in range(batch):
                if predicted[i] != 0:
                    self.ood_count += 1
                    with torch.no_grad():
                        noise = torch.randn(1, self.latent_dim).to(self.device)
                        all_labels = list(range(self.num_classes))  # 数据集共有10个类别
                        #all_labels.remove(known_label.item())  # 移除已知标签
                        random_label = torch.tensor(random.choice(all_labels)).unsqueeze(0).to(self.device)
                        gen_class_probs = self.generator(random_label, noise)
                        gen_probs_list.append(gen_class_probs.squeeze(0))
                else:
                    self.mis_correct_count += 1
                    gen_probs_list.append(y[i])
        gen_probs_tensor = torch.stack(gen_probs_list)
        gen_probs_tensor = {0.0: gen_probs_tensor}
        # print(gen_probs_tensor)
        return gen_probs_tensor

    def get_stats(self):

        print("mis", self.mis_correct_count)
        return self.mis_correct_count


    def reset_stats(self):
        self.ood_count = 0
        self.input_count = 0
        self.mis_correct_count = 0