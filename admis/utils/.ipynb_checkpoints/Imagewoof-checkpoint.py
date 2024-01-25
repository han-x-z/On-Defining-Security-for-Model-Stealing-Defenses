import os.path as osp
import numpy as np
from torchvision.datasets import ImageFolder

class ImageWoof(ImageFolder):
    test_frac = 0.2  # 用于测试的数据集比例

    def __init__(self, train=True, transform=None, target_transform=None, root="./data"):
        root = osp.join(root, 'imagewoof2-160')  # 修改为imagewoof的路径
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, '下载链接'  # 提供imagewoof数据集的下载链接
            ))

        # 初始化ImageFolder
        super().__init__(root=osp.join(root, 'train'), transform=transform,
                         target_transform=target_transform)
        self.root = root

        self.partition_to_idxs = self.get_partition_to_idxs()
        self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        # 修剪样本，只包含所需的训练/测试部分
        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))

    def get_partition_to_idxs(self):
        partition_to_idxs = {'train': [], 'test': []}
        prev_state = np.random.get_state()
        np.random.seed(123)

        idxs = np.arange(len(self.samples))
        n_test = int(self.test_frac * len(idxs))
        test_idxs = np.random.choice(idxs, replace=False, size=n_test).tolist()
        train_idxs = list(set(idxs) - set(test_idxs))

        partition_to_idxs['train'] = train_idxs
        partition_to_idxs['test'] = test_idxs

        np.random.set_state(prev_state)
        return partition_to_idxs
