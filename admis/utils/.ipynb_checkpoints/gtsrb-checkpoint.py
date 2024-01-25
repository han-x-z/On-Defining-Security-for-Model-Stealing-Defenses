
import os.path as osp
from torchvision.datasets import ImageFolder


class GTSRB(ImageFolder):
    def __init__(self, train=True, transform=None, target_transform=None, root="./data/gtsrb"):
        root = osp.join(root, 'GTSRB', 'Final_Training')
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset'
            ))

        # Initialize ImageFolder
        super().__init__(root=osp.join(root, 'Images'), transform=transform,
                         target_transform=target_transform)

        self.root = root
        trainning_size = len(self.samples)
        self.read_test(osp.join('./data/gtsrb', 'GTSRB', 'Final_Test', 'Images'))

        self.partition_to_idxs = self.get_partition_to_idxs(trainning_size)
        self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        # Prune (self.imgs, self.samples to only include examples from the required train/test partition
        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))

    def read_test(self, folder):
        with open(osp.join(folder, "GT-final_test.csv")) as f:
            f.readline()  # 跳过标题行
            for line in f:
                parts = line.strip().split(";")
                # 假设最后一个字段是标签
                label = parts[-1]
                # 图像路径是第一个字段
                image = parts[0]
                path = osp.join(folder, image)
                self.samples.append((path, int(label)))
                self.targets.append(int(label))

    def get_partition_to_idxs(self, training_size):
        partition_to_idxs = {
            'train': [],
            'test': []
        }
        for i in range(training_size):
            partition_to_idxs['train'].append(i)
        for i in range(training_size, len(self.samples)):
            partition_to_idxs['test'].append(i)
        return partition_to_idxs
