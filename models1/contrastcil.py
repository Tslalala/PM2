import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import timm
from collections import OrderedDict
from transformers import ViTForImageClassification
'''
    还没写
    想法: 借助对比学习损失，并且在VPT训练时，损失再加一个维护与其他proto距离的一个损失
    head分类头直接不要了, 拿对比损失做聚类
'''

from .base import BaseLeaner
from convs.vpt import build_promptmodel

class Learner(BaseLeaner):
    def __init__(self, args):
        super().__init__(args)
        self.prompt_pool = []
        self.classes_per_task = []
        self.args = args
        self._network_prompt = build_promptmodel(modelname="vit_base_patch16_224_in21k", Prompt_Token_num=self.args["Prompt_Token_num"],
                                                 VPT_type=self.args["VPT_type"], args=self.args, new_classes=5)

    def incremental_train(self, data_manager):
        self._cur_task = self._cur_task + 1
        self.classes_per_task.append(self._cur_task)
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        # 数据包构建
        batch_size = 128
        train_indices = np.arange(self._known_classes, self._total_classes)
        test_indices = np.arange(0, self._total_classes)

        train_dataset = data_manager.get_dataset(indices=train_indices, source="train", mode="train", )
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        test_dataset = data_manager.get_dataset(indices=test_indices, source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        train_dataset_for_protonet = data_manager.get_dataset(indices=train_indices, source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=batch_size, shuffle=True, num_workers=0)

        # 只研究前三个数据包
        if self._known_classes == 0:
            self.train_loader0 = self.train_loader_for_protonet
        elif self._known_classes == self.args["init_cls"]:
            self.train_loader1 = self.train_loader_for_protonet
        elif self._known_classes == self.args["init_cls"] + self.args["increment"]:
            self.train_loader2 = self.train_loader_for_protonet

        # 训练
        self._train()

    def _train(self,):

        # if self._known_classes == 0:
        if self._known_classes != -1:
            self._network = build_promptmodel(modelname="vit_base_patch16_224_in21k", Prompt_Token_num=self.args["Prompt_Token_num"],
                                              VPT_type=self.args["VPT_type"], args=self.args, new_classes=self._total_classes)
            self._network.to(self._device)
            self.train_vpt()
            self._network.eval()

        # 推理
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.train_loader_for_protonet)):
                (_, data, label) = batch
                data = data.to(self._device)
                # label = label.to(self._device)

                # 获取proto的核心, model.convnet(), 来自BaseNet类
                embedding = self._network.forward_features_(data)
                # print('embedding.shape', embedding.shape)
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
            proto = embeddings.mean(0)
            # print('proto_dim:', proto.shape)
            feature_proto_list.append(proto)

        # 传递proto_list
        if self._known_classes == 0:
            self.feature_proto_list = feature_proto_list
        else:
            self.feature_proto_list = self.feature_proto_list + feature_proto_list

    def train_vpt(self,):

        # Freeze the parameters for ViT.
        total_params = sum(p.numel() for p in self._network.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        # 参数初始化
        if self._known_classes != 0:
            self._network.load_prompt(self.prompt_pool[-1])
            self._network.to(self._device)

        # if some parameters are trainable, print the key name and corresponding parameter number
        if total_params != total_trainable_params:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    print(name, param.numel())

        # optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=1e-2, weight_decay=1e-4)
        optimizer = optim.SGD(self._network.parameters(), momentum=0, lr=self.args["lr_prompt"], weight_decay=5e-4)
        loss_fn = nn.CrossEntropyLoss()

        # 训练VPT
        # prog_bar = tqdm(range(self.args["tuned_epoch"]))
        # for _, epoch in enumerate(prog_bar):
        for _, epoch in enumerate(range(self.args["tuned_epoch"])):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(self.train_loader_for_protonet):
                inputs, targets = inputs.to(self._device), targets.to(self._device).long()
                logits = self._network.forward(inputs)
                # print('logits.shape, targets.shape', logits.shape, targets.shape)
                # targets = targets - self._known_classes
                # print('targets:', targets)
                loss = loss_fn(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            print('losses:', losses)

            correct = correct.cpu().data.numpy() if correct.is_cuda else correct.data.numpy()
            train_acc = np.around(correct * 100 / total, decimals=2)
            print(f'epoch{epoch} train_acc:', train_acc)
            if (losses < 2.30) or (train_acc > 97.80):
                break

        # 保存prompt_token
        self.prompt_pool.append(self._network.obtain_prompt())

    def eval_task(self,):
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(tqdm(self.test_loader)):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                # 初始化一个列表来存储每个输出的最佳预测和对应的value
                best_predicts = [None] * len(inputs)
                best_values = [-float('inf')] * len(inputs)

                for prompt_tokens in self.prompt_pool:
                    self._network_prompt.load_prompt(prompt_tokens)
                    self._network_prompt.to(self._device)
                    outputs = self._network_prompt.forward_features_(inputs)

                    for idx, test_output in enumerate(outputs):
                        predict, value = self.classify_with_proto_prompt(test_output, proto_list=self.feature_proto_list)

                        # 如果当前value大于已记录的最大value，则更新记录
                        if value > best_values[idx]:
                            best_values[idx] = value
                            best_predicts[idx] = predict

                # 将最佳预测添加到y_pred列表中，并将对应的target添加到y_true列表中
                for idx, predict in enumerate(best_predicts):
                    # print('predict:', predict[0])
                    y_pred.append(predict)
                    y_true.append(targets[idx].item())

        return y_pred, y_true

    def eval_cur_task_on_train_loader(self, train_loader):
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(tqdm(train_loader)):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network.forward_features_(inputs)

                for _, test_output in enumerate(outputs):
                    predict = self.classify_with_proto(test_output, proto_list=self.feature_proto_list[self._known_classes : self._total_classes])
                    predict = [predict[0] + self._known_classes]
                    # print('predict:', predict)
                    y_pred.append(predict)
                    y_true.append(targets[_].item())

        return y_pred, y_true

    def classify_with_proto(self, test_output, proto_list):
        top_num = 2
        test_output = test_output.cpu()

        # SimpleCIL分类
        test_img_distance_coordinate_incre = []
        for i in range(len(proto_list)):
            # L2距离
            distance = torch.sum((test_output - proto_list[i]) ** 2)
            test_img_distance_coordinate_incre.append(1 / distance)

        test_img_distance_coordinate_incre = torch.tensor(test_img_distance_coordinate_incre)
        # print('test_img_distance_coordinate_incre', test_img_distance_coordinate_incre)

        # 如果SimpleCIL分类拥有较高置信度
        value_, index_ = torch.topk(test_img_distance_coordinate_incre, k=top_num, dim=0, largest=True, sorted=True)
        test_img_coordinate_incre = index_.numpy().tolist()

        predict = test_img_coordinate_incre

        return predict

    def classify_with_proto_prompt(self, test_output, proto_list):
        top_num = 2
        test_output = test_output.cpu()

        # SimpleCIL分类
        test_img_distance_coordinate_incre = []
        for i in range(len(proto_list)):
            # L2距离
            distance = torch.sum((test_output - proto_list[i]) ** 2)
            test_img_distance_coordinate_incre.append(1 / distance)

        test_img_distance_coordinate_incre = torch.tensor(test_img_distance_coordinate_incre)

        # 如果SimpleCIL分类拥有较高置信度
        value_, index_ = torch.topk(test_img_distance_coordinate_incre, k=top_num, dim=0, largest=True, sorted=True)
        test_img_coordinate_incre = index_.numpy().tolist()

        predict = test_img_coordinate_incre

        return predict, value_[0].numpy()