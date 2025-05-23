import numpy as np
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import random


class StrongAugmentation:
    def __init__(self):
        # 核心参数
        self.prob = 0.8
        self.elastic_alpha = 120

    def __call__(self, img):
        # 几何破坏
        img = self.elastic_transform(img) if random.random() < self.prob else img

        # 颜色攻击
        img = self.channel_attack(img)

        # 物理降级
        img = self.motion_blur(img) if random.random() < 0.7 else img
        img = self.random_occlusion(img)

        return img

    def elastic_transform(self, img):
        """弹性变形"""
        img = np.array(img)
        shape = img.shape
        dx = np.random.randint(-self.elastic_alpha, self.elastic_alpha, shape[:2])
        dy = np.random.randint(-self.elastic_alpha, self.elastic_alpha, shape[:2])
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        return Image.fromarray(
            cv2.remap(img, (x + dx).astype(np.float32), (y + dy).astype(np.float32),
                      interpolation=cv2.INTER_LINEAR)
        )

    def channel_attack(self, img):
        """通道攻击"""
        if random.random() < 0.6:
            channels = list(img.split())
            random.shuffle(channels)
            img = Image.merge('RGB', channels)
        img = ImageEnhance.Color(img).enhance(random.uniform(0.1, 2.5))
        return img

    def motion_blur(self, img):
        """运动模糊"""
        angle = random.randint(0, 180)
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 5))).filter(
            ImageFilter.Kernel((15, 15),
                               ImageFilter.RankFilter(15, 5).kernel[angle % 2])
        )

    def random_occlusion(self, img):
        """随机遮挡"""
        w, h = img.size
        for _ in range(random.randint(1, 3)):
            x1 = random.randint(0, w // 2)
            y1 = random.randint(0, h // 2)
            x2 = random.randint(x1 + 50, w)
            y2 = random.randint(y1 + 50, h)
            img.paste((random.randint(0, 255),) * 3, (x1, y1, x2, y2))
        return img

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

def build_transform(is_train=True, is_StrongAugmentation=False):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)

        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())

    if is_StrongAugmentation:
        t.append([
            StrongAugmentation(),
            transforms.RandomHorizontalFlip(p=0.8),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomErasing(p=0.8, scale=(0.1, 0.3)),
            transforms.ToTensor(),
        ])

    # return transforms.Compose(t)
    return t

def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

# 以下为具体数据集获取即预处理

# CIFAR100
class iCIFAR100(iData):
    use_path = False

    train_trsf = build_transform(is_train=True)
    test_trsf = build_transform(is_train=False)
    common_trsf = [transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))]

    # 未打乱的class_order
    class_order = np.arange(100).tolist()

    def download_data(self):
        # 此时无需transforms
        train_dataset = datasets.cifar.CIFAR100('H:/DATA4DL/cifar', train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100('H:/DATA4DL/cifar', train=False, download=True)

        # 获取data与targets
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)

class iCIFAR224(iData):
    use_path = False

    train_trsf = build_transform(is_train=True)
    test_trsf = build_transform(is_train=False)
    strong_trsf = build_transform(is_StrongAugmentation=True)
    common_trsf = []

    # 未打乱的class_order
    class_order = np.arange(100).tolist()

    def download_data(self):
        # 此时无需transforms
        train_dataset = datasets.cifar.CIFAR100('E:/DATA4DL/cifar', train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100('E:/DATA4DL/cifar', train=False, download=True)

        # 获取data与targets
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)

class CUB(iData):
    use_path = True

    train_trsf = build_transform(is_train=True)
    test_trsf = build_transform(is_train=False)
    strong_trsf = build_transform(is_StrongAugmentation=True)
    common_trsf = []

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = 'E:/DATA4DL/CUB/train/'
        test_dir = 'E:/DATA4DL/CUB/test/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetA(iData):
    use_path = True

    train_trsf = build_transform(is_train=True)
    test_trsf = build_transform(is_train=False)
    strong_trsf = build_transform(is_StrongAugmentation=True)
    common_trsf = []

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = 'E:/DATA4DL/ImageNet-A/train/'
        test_dir = 'E:/DATA4DL/ImageNet-A/test/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetR(iData):
    use_path = True

    train_trsf = build_transform(is_train=True)
    test_trsf = build_transform(is_train=False)
    strong_trsf = build_transform(is_StrongAugmentation=True)
    common_trsf = []

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = 'E:/DATA4DL/ImageNet-R/train/'
        test_dir = 'E:/DATA4DL/ImageNet-R/test/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class objectnet(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/objectnet/train/"
        test_dir = "./data/objectnet/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class omnibenchmark(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []

    class_order = np.arange(300).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/omnibenchmark/train/"
        test_dir = "./data/omnibenchmark/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class vtab(iData):
    use_path = True

    train_trsf = build_transform(is_train=True)
    test_trsf = build_transform(is_train=False)
    strong_trsf = build_transform(is_StrongAugmentation=True)
    common_trsf = []

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = 'E:/DATA4DL/vtab/train/'
        test_dir = 'E:/DATA4DL/vtab/test/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)