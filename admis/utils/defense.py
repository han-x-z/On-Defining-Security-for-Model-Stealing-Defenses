import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import uniform
import torch.nn as nn
import random
from torchvision import models

def compute_hellinger(y_a, y_b):
    """
    :param y_a: n x K dim
    :param y_b: n x K dim
    :return: n dim vector of hell dist between elements of y_a, y_b
    """
    diff = torch.sqrt(y_a) - torch.sqrt(y_b)
    sqr = torch.pow(diff, 2)
    hell = torch.sqrt(0.5 * torch.sum(sqr, dim=1))
    return hell.cpu().detach().numpy()


class selectiveMisinformation:
    def __init__(
        self, model_mis, delta_list, num_classes=10, rand_fhat=False, use_adaptive=True
    ):
        self.delta_list = delta_list
        self.input_count = 0
        self.ood_count = {}
        self.num_classes = num_classes
        self.correct_class_rank = np.zeros(self.num_classes)
        self.mis_correct_count = 0
        self.hell_dist = {}
        self.max_probs = {}
        self.alpha_vals = {}
        self.reset_stats()
        self.model_mis = model_mis
        self.use_adaptive = use_adaptive
        self.rand_fhat = rand_fhat

    def __call__(self, x, y):
        probs = y  # batch x 10
        probs_max, probs_max_index = torch.max(probs, dim=1)  # batch
        batch = probs_max.size(0)
#        print(batch)
        self.input_count += batch
        y_mis = self.model_mis(x)
        y_mis = F.softmax(y_mis, dim=1)
        probs_mis_max, probs_mis_max_index = torch.max(y_mis, dim=1)  # batch
        self.mis_correct_count += (probs_mis_max_index == probs_max_index).sum().item()
        y_mis_dict = {}
        for delta in self.delta_list:
            y_mis = y_mis.detach()
            if self.use_adaptive:
                h = 1 / (1 + torch.exp(-10000 * (delta - probs_max.detach())))
            else:
                h = delta * torch.ones_like(probs_max.detach())

            h = h.unsqueeze(dim=1).float()
            mask_ood = probs_max <= delta
            if delta not in self.ood_count:
                self.ood_count[delta] = 0
            self.ood_count[delta] += np.sum(mask_ood.cpu().detach().numpy())
            y_mis_dict[delta] = ((1.0 - h) * y) + (h * y_mis.float())
            probs_mis_max, _ = torch.max(y_mis_dict[delta], dim=1)
            self.max_probs[delta].append(probs_mis_max.cpu().detach().numpy())
            self.alpha_vals[delta].append(h.squeeze(dim=1).cpu().detach().numpy())
            hell = compute_hellinger(y_mis_dict[delta], y)
            self.hell_dist[delta].append(hell)
            print(y_mis_dict)
        return y_mis_dict

    def get_stats(self):
        rejection_ratio = {}
        for delta in self.delta_list:
            rejection_ratio[delta] = float(self.ood_count[delta]) / float(
                self.input_count
            )
            print("Delta: {} Rejection Ratio: {}".format(delta, rejection_ratio[delta]))
            self.hell_dist[delta] = np.array(np.concatenate(self.hell_dist[delta]))
            self.max_probs[delta] = np.array(np.concatenate(self.max_probs[delta]))
            self.alpha_vals[delta] = np.array(np.concatenate(self.alpha_vals[delta]))
        print(
            "miss_correct_ratio: ",
            float(self.mis_correct_count) / float(self.input_count),
        )
        np.savez_compressed("./logs/hell_dist_sm", a=self.hell_dist)
        np.savez_compressed("./logs/max_probs", a=self.max_probs)
        np.savez_compressed("./logs/alpha_vals", a=self.alpha_vals)
        return rejection_ratio

    def reset_stats(self):
        for delta in self.delta_list:
            self.ood_count[delta] = 0
            self.hell_dist[delta] = []
            self.max_probs[delta] = []
            self.alpha_vals[delta] = []
        self.input_count = 0
        self.mis_correct_count = 0
        self.correct_class_rank = np.zeros(self.num_classes)


class noDefense:
    def __call__(self, x, y):
        y_noDef = {}
        y_noDef[0] = y
        return y_noDef

    def print_stats(self):
        return

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

class selfGenerator:
    def __init__(
        self, delta_list, num_classes=10, 
    ):
        self.delta_list = delta_list
        self.input_count = 0
        self.ood_count = {}
        self.num_classes = num_classes
        self.latent_dim = 10  # 定义 latent_dim 的值为 100
        self.mis_correct_count = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else False)
        self.generator = Generator(self.latent_dim, self.num_classes).to(self.device)
            
    def __call__(self, x, y):
        
        
        probs = y  # batch x 10
#        print(probs)
        batch = probs.size(0)
#        print(batch)
        self.input_count += batch
#        print(batch)
        # 加载并评估生成器
        self.generator.load_state_dict(torch.load('./sampler/res/image/1/generator60.pth')) #TODO
        self.generator.eval()
        delta = 0.8    #TODO
        gen_probs = {} 
        gen_probs[0] = [None] * batch
        num_from_generator = int(delta * batch)
#        print(num_from_generator)
        with torch.no_grad():
            gen_probs_list = []  # 用于收集所有生成的输出或模型输出的列表
            for i in range(batch):
                if i < num_from_generator:
                    # 从生成器获取输出
                    noise = torch.randn(1, self.latent_dim).to(self.device)
                    all_labels = list(range(self.num_classes))
                    random_label = torch.tensor(random.choice(all_labels)).unsqueeze(0).to(self.device)
                    gen_class_probs = self.generator(random_label, noise)
                    gen_probs_list.append(gen_class_probs.squeeze(0))  # 将生成的输出添加到列表中
                else:
                    gen_probs_list.append(y[i])  # 将模型的输出添加到列表中

    # 将列表转换为二维张量
        gen_probs_tensor = torch.stack(gen_probs_list)
        gen_probs_tensor = {0.0: gen_probs_tensor} 
        return gen_probs_tensor


class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 64 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)
        
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
        self.BinaryClassifier = BinaryClassifier().to(self.device)
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
        self.generator.load_state_dict(torch.load('./sampler/res/mnist/1/generator130.pth'))
        self.generator.eval()
        #TODO
#        self.model.classifier[6] = torch.nn.Linear(4096, 3).to(self.device)        
#        self.model.load_state_dict(torch.load('./admis/utils/gtsrb/attackgtsrb1.pth'))
        # 修改全连接层以进行3分类
        #num_ftrs = self.model1.fc.in_features  
        #self.model1.fc = nn.Linear(num_ftrs, 3).to(self.device)  # 3分类
        #self.model1.load_state_dict(torch.load('./admis/utils/image/attackimage1.pth'))
        self.MultiClassifier.load_state_dict(torch.load('./detector/mnist/attackmnist3.pth'))
        self.MultiClassifier.eval()
        #TODO
        gen_probs = {}
        gen_probs[0] = [None] * batch
        with torch.no_grad():
            gen_probs_list = []
#            outputs = self.model(x).to(self.device)
            outputs = self.MultiClassifier(x)
            #outputs = self.model1(x).to(self.device)
            #TODO
            _, predicted = torch.max(outputs, 1)
            
#            print(predicted)
            for i in range(batch):
                
                if predicted[i] != 0 :
                    self.ood_count += 1
                    with torch.no_grad():
                        noise = torch.randn(1, self.latent_dim).to(self.device)
                        all_labels = list(range(self.num_classes))  # 数据集共有10个类别
    #                    all_labels.remove(known_label.item())  # 移除已知标签
                        random_label = torch.tensor(random.choice(all_labels)).unsqueeze(0).to(self.device)
                        gen_class_probs = self.generator(random_label, noise)
                        gen_probs_list.append(gen_class_probs.squeeze(0))
                else:
                    self.mis_correct_count+= 1
                    gen_probs_list.append(y[i])
        gen_probs_tensor = torch.stack(gen_probs_list)
        gen_probs_tensor = {0.0: gen_probs_tensor} 
#        print(gen_probs_tensor)
        return gen_probs_tensor            

    def get_stats(self):
        rejection_ratio = 0
        rejection_ratio = float(self.ood_count) / float(self.input_count)
        print("Rejection Ratio: {}".format(rejection_ratio))
        print(
            "correct_ratio: ",
            float(self.mis_correct_count) / float(self.input_count),
        )
        print(
            "mis_correct_count: ",self.mis_correct_count
        )
        return rejection_ratio

    def reset_stats(self):
        self.ood_count = 0
        self.input_count = 0
        self.mis_correct_count = 0