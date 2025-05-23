import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

import utils.toolkits as toolkits
from utils.toolkits import seed_set

seed_set()


# 封装数据集
class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            path = self.images[idx]
            with open(path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
            image = self.trsf(img)
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


# 数据管理
class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args):
        self.args = args
        self.dataset_name = dataset_name
        self.shuffle = shuffle

        # 加载并划分数据集
        self._setup_data(dataset_name, shuffle)

        self._increments = toolkits.get_increments_list(init_cls, increment, self._class_order)

    # 加载并划分数据集, 得到data与targets
    def _setup_data(self, dataset_name, shuffle):
        # idata=iCIFAR100
        idata = toolkits.get_idata(dataset_name, self.args)
        idata.download_data()

        # 传递idata中的data与targets
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._strong_trsf = idata.strong_trsf
        self._common_trsf = idata.common_trsf

        # 是否打乱class_order
        if shuffle:
            order = [i for i in range(len(np.unique(self._train_targets)))]
            order = np.random.permutation(len(order)).tolist()
            self._class_order = order
        else:
            order = idata.class_order
            self._class_order = order

        # 重新排序targets
        self.train_targets = [self._class_order.index(x) for x in self._train_targets]
        self.test_targets = [self._class_order.index(x) for x in self._test_targets]

    # task中包含class数目
    def get_task_size(self, task_id):
        task_size = self._increments[task_id]
        return task_size

    def get_dataset(self, indices, source, mode):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose([
                *self._test_trsf,
                transforms.RandomHorizontalFlip(p=1.0),
                *self._common_trsf,
            ])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        elif mode == "strong":
            trsf = transforms.Compose([*self._strong_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        # 构建数据包
        data, targets = [], []
        for idx in indices:
            low_range = idx
            high_range = idx + 1
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            # print('len(idxes)', len(idxes))

            class_data = x[idxes]
            class_targets = y[idxes]
            data.append(class_data)
            targets.append(class_targets)
        data, targets = np.concatenate(data), np.concatenate(targets)

        return DummyDataset(data, targets, trsf, self.use_path)

