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

from .base import BaseLeaner


class Learner(BaseLeaner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self._network.eval()
        self.outputs_list_mean_ImageNet_val200 = torch.load('../PTM_with_coordinate_proto/utils/outputs_list_mean_ImageNet_val200.pth')

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
        constantcoordinate_proto_list = []
        for class_index in class_list:
            # print('Replacing...',class_index)
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embeddings = embedding_list[data_index]

            # 这里开始与simplecil不同
            embeddings_coordinate_200 = []
            for output_that in embeddings:

                embedding_coordinate_200 = []
                for outputs_list_mean_ImageNet_val200 in self.outputs_list_mean_ImageNet_val200:
                    distance = torch.sum((output_that - outputs_list_mean_ImageNet_val200) ** 2)
                    embedding_coordinate_200.append(distance ** -2)     # 这里的指数可以更改
                embedding_coordinate_200 = torch.tensor(embedding_coordinate_200)
                embeddings_coordinate_200.append(embedding_coordinate_200)

            embeddings_coordinate_200 = torch.stack(embeddings_coordinate_200)
            proto = embeddings_coordinate_200.mean(0)
            constantcoordinate_proto_list.append(proto)

        # 传递proto_list
        if self._known_classes == 0:
            self.constantcoordinate_proto_list = constantcoordinate_proto_list
        else:
            self.constantcoordinate_proto_list = self.constantcoordinate_proto_list + constantcoordinate_proto_list

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
        top_num = 5
        test_output = test_output.cpu()
        predict = [-1 for _ in range(top_num)]

        test_embed_coordinate = []
        for outputs_list_mean_ImageNet_val200 in self.outputs_list_mean_ImageNet_val200:
            distance = torch.sum((test_output - outputs_list_mean_ImageNet_val200) ** 2)
            test_embed_coordinate.append(distance ** -2)    # 这里的指数可以更改
        test_embed_coordinate = torch.tensor(test_embed_coordinate)

        distance_with_protos = []
        for proto in self.constantcoordinate_proto_list:
            distance = torch.sum((test_embed_coordinate - proto) ** 2)
            distance_with_protos.append(1/distance)
        distance_with_protos = torch.tensor(distance_with_protos)


        value_, index_ = torch.topk(distance_with_protos, k=top_num, dim=0, largest=True, sorted=True)
        test_img_coordinate_incre = index_.numpy().tolist()
        top0_index = test_img_coordinate_incre[0]
        top1_index = test_img_coordinate_incre[1]
        top2_index = test_img_coordinate_incre[2]

        # 条件判断是否仅使用SimpleCIL分类
        if distance_with_protos[top0_index] > 1.0 * distance_with_protos[top1_index]:
            predict = test_img_coordinate_incre

        # predict为一list包含前top_num个预测结果
        return predict
