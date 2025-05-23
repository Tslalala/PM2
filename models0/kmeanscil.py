import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import timm
from sklearn.cluster import KMeans

from .base import BaseLeaner


class Learner(BaseLeaner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self._network.eval()

    def incremental_train(self, data_manager):
        self._cur_task = self._cur_task + 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        # 数据包构建
        batch_size = 124
        train_indices = np.arange(self._known_classes, self._total_classes)
        test_indices = np.arange(0, self._total_classes)

        train_dataset = data_manager.get_dataset(indices=train_indices, source="train", mode="train", )
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        test_dataset = data_manager.get_dataset(indices=test_indices, source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        train_dataset_for_protonet = data_manager.get_dataset(indices=train_indices, source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=batch_size, shuffle=True, num_workers=0)

        # 训练
        # self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        self._train()

    def _train(self,):
        self._network.to(self._device)

        # 推理
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.train_loader_for_protonet)):
                (_, data, label) = batch
                data = data.to(self._device)
                # label = label.to(self._device)

                # 获取proto的核心, model.convnet(), 来自BaseNet类
                embedding = self._network(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        # NCM, 对class’s features取mean
        class_list = np.unique(label_list)
        feature_proto_list = []
        for class_index in class_list:
            # print('Replacing...',class_index)
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embeddings = embedding_list[data_index]

            # k-means
            embeddings_np = np.array([embed.numpy() for embed in embeddings])
            kmeans = KMeans(n_clusters=3, random_state=0)
            kmeans.fit(embeddings_np)
            centers = kmeans.cluster_centers_
            proto = [torch.tensor(center) for center in centers]
            feature_proto_list.append(proto)

        # 传递proto_list
        if self._known_classes == 0:
            self.feature_proto_list = feature_proto_list
        else:
            self.feature_proto_list = self.feature_proto_list + feature_proto_list

    def eval_task(self):
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(tqdm(self.test_loader)):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)

                for _, test_output in enumerate(outputs):
                    predict = self.classify_with_proto(test_output)
                    y_pred.append(predict)
                    y_true.append(targets[_].item())

        return y_pred, y_true

    def classify_with_proto(self, test_output):
        top_num = 3
        test_output = test_output.cpu()

        # CrazyCIL分类
        # 找到每个特征集中的最小距离
        min_distances = []
        for protos in self.feature_proto_list:
            protos = torch.stack(protos)    # 比crazycil多一个连结操作

            # 计算 test_output 与当前特征原型集的距离
            distances = torch.sum((test_output - protos) ** 2, dim=1)

            # 找到最小距离
            min_distance = torch.min(distances)
            min_distances.append(min_distance)

        # 计算倒数并存储结果
        test_img_distance_coordinate_incre = [1 / d for d in min_distances]
        test_img_distance_coordinate_incre = torch.tensor(test_img_distance_coordinate_incre)

        # 如果SimpleCIL分类拥有较高置信度
        value_, index_ = torch.topk(test_img_distance_coordinate_incre, k=top_num, dim=0, largest=True, sorted=True)
        predict = index_.numpy().tolist()

        return predict
