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
from collections import OrderedDict
from transformers import ViTForImageClassification

from .base import BaseLeaner
from convs.vpt import build_promptmodel

class Learner(BaseLeaner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = build_promptmodel(modelname="vit_base_patch16_224_in21k",  Prompt_Token_num=args["Prompt_Token_num"], VPT_type=args["VPT_type"], args=args)
        self._network.eval()

    def incremental_train(self, data_manager):
        self._cur_task = self._cur_task + 1
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
        # self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        self._train()

    def _train(self,):
        self._network.to(self._device)

        # if self._known_classes == 0:
        if self._known_classes != -1:
            # self._network = build_promptmodel(modelname="vit_base_patch16_224_in21k", Prompt_Token_num=self.args["Prompt_Token_num"], VPT_type=self.args["VPT_type"], args=self.args)
            self.train_vpt()
            self._network.eval()

    def train_vpt(self,):

        # Freeze the parameters for ViT.
        total_params = sum(p.numel() for p in self._network.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self._network.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        # if some parameters are trainable, print the key name and corresponding parameter number
        if total_params != total_trainable_params:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    print(name, param.numel())

        # optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=1e-2, weight_decay=1e-4)
        optimizer = optim.SGD(self._network.parameters(), momentum=0, lr=1e-2, weight_decay=0)
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
                targets = targets - self._known_classes
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
            print('train_acc:', train_acc)

    def eval_task(self,):
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(tqdm(self.test_loader)):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network.forward(inputs)
                for _, test_output in enumerate(outputs):
                    value, predict = torch.max(test_output, dim=0)
                    y_pred.append([predict])
                    y_true.append(targets[_].item())

        return y_pred, y_true

    def eval_cur_task_on_train_loader(self, train_loader):
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(tqdm(train_loader)):
            inputs = inputs.to(self._device)
            targets = targets - self._known_classes
            with torch.no_grad():
                outputs = self._network.forward(inputs)
                for _, test_output in enumerate(outputs):
                    value, predict = torch.max(test_output, dim=0)
                    y_pred.append([predict])
                    y_true.append(targets[_].item())

        return y_pred, y_true

