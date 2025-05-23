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
        self._network = timm.create_model(self.args["pretrained_model"], pretrained=True, num_classes=0)
        self._network.eval()

    def incremental_train(self, data_manager):
        self._cur_task = self._cur_task + 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        # 数据包构建
        batch_size = 512
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
        self.embedding_list = torch.cat(embedding_list, dim=0)
        self.label_list = torch.cat(label_list, dim=0)

        # NCM, 对class’s features取mean
        self.class_list = np.unique(self.label_list)
        self.feature_proto_list = []
        for class_index in self.class_list:
            # print('Replacing...',class_index)
            data_index = (self.label_list == class_index).nonzero().squeeze(-1)
            embeddings = self.embedding_list[data_index]

            # k-means产生几个proto用以初始化
            embeddings_np = np.array([embed.numpy() for embed in embeddings])
            kmeans = KMeans(n_clusters=self.args["n_clusters"], n_init='auto', random_state=0)
            kmeans.fit(embeddings_np)
            centers = kmeans.cluster_centers_
            proto = [torch.tensor(center) for center in centers]
            self.feature_proto_list.append(proto)

        # 梯度更新以获取gdproto_list
        gdproto_list = self.gradient_descent_proto()
        # gdproto_list = [proto.to(self._device) for proto in self.feature_proto_list]    # 选用这个则为simplecil

        # 传递proto_list
        if self._known_classes == 0:
            self.gdproto_list = gdproto_list
        else:
            for param in gdproto_list:
                self.gdproto_list.append(param)

    def gradient_descent_proto(self):
        print('gradient descent proto...')
        proto_num = len(self.feature_proto_list)

        # 创建一个空的 nn.ParameterList
        gdproto_list = nn.ModuleList()  # 使用 ModuleList 来表示一个两层的参数结构

        # 填充 gdproto_list
        for proto_layer in self.feature_proto_list:
            # 对于 feature_proto_list 的每一层，创建一个新的 nn.ParameterList 并将其添加到 gdproto_list 中
            proto_layer_list = nn.ParameterList([
                nn.Parameter(proto.clone().detach().to(self._device), requires_grad=True) for proto in proto_layer
            ])
            gdproto_list.append(proto_layer_list)

        # 优化器与损失函, 目前最适配的lr=1e2, epochs<=5, n_clusters=100
        optimizer = optim.SGD(gdproto_list.parameters(), lr=self.args["lr"])
        loss_fn = nn.MSELoss()

        n_epochs = self.args["n_epochs"]
        # for epoch in tqdm(range(n_epochs)):
        for epoch in range(n_epochs):
            total_loss = 0.
            for class_index in self.class_list:
                optimizer.zero_grad()

                # 确保 class_index_tensor 为浮点型并在设备上
                class_index_tensor = torch.tensor(class_index - self._known_classes, device=self._device, dtype=torch.int64)    # 要减去_known_classes否则范围不对
                class_index_one_hot = F.one_hot(class_index_tensor, num_classes=proto_num).to(dtype=torch.float32)

                data_index = (self.label_list == class_index).nonzero().squeeze(-1)
                embeddings = self.embedding_list[data_index]

                # loss = torch.tensor(0., device=self._device, requires_grad=True)    # 初始化
                for embed in embeddings:
                    # 获取预测结果，确保它是一个张量
                    predict = self.classify_with_proto(test_output=embed, proto_list=gdproto_list)

                    # 计算损失
                    loss = loss_fn(predict, class_index_one_hot)

                    # 反向传播
                    # print('loss:', loss.item())
                    total_loss = total_loss + loss
            print('total_loss:', total_loss.item())
            total_loss.backward()
            optimizer.step()

        return gdproto_list

    def eval_task(self, ):
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(tqdm(self.test_loader)):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)

                for _, test_output in enumerate(outputs):
                    predict = self.classify_with_proto(test_output, proto_list=self.gdproto_list)
                    value, predict = torch.topk(predict, k=2, dim=0, largest=True, sorted=True)
                    y_pred.append(predict.cpu().numpy().tolist())
                    y_true.append(targets[_].item())

        return y_pred, y_true

    def classify_with_proto(self, test_output, proto_list):
        if proto_list is None:
            proto_list = self.gdproto_list

        test_output = test_output.to(self._device)  # 确保 test_output 已在设备上

        # CrazyCIL分类
        # 找到每个特征集中的最小距离
        min_distances = []
        for protos in proto_list:
            protos = [param for param in protos]
            protos = torch.stack(protos)  # 比crazycil多一个连结操作

            # 计算 test_output 与当前特征原型集的距离
            distances = torch.sum((test_output - protos) ** 2, dim=1)

            # 找到最小距离
            min_distance = torch.min(distances)
            # print('min_distance.requires_grad:', min_distance.requires_grad)
            min_distances.append(min_distance)

        # 计算倒数并存储结果
        test_img_distance_coordinate_incre = [1 / d for d in min_distances]
        test_img_distance_coordinate_incre = torch.stack(test_img_distance_coordinate_incre)

        # 使用 softmax以不丢失计算图
        index_ = F.softmax(test_img_distance_coordinate_incre / 1e-4, dim=0)
        predict = index_  # 这里的predict将继续在计算图中

        return predict
