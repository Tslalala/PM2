import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset

from convs.losses import NCMLoss
import timm
from convs.mine11 import Mine11
import utils.toolkits as toolkits


class Learner:
    def __init__(self, args):
        self.adapter_ = None
        self.cur_classes = None
        self.prototypes = None
        self.args = args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self.classes_per_task = []
        self.feature_proto_list = None
        self.adapter_pool = []
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._old_network = self.call_model()

    def after_task(self):
        self._known_classes = self._total_classes

    def call_model(self):
        return Mine11()

    def incremental_train(self, data_manager):
        self._cur_task = self._cur_task + 1
        cur_classes_num = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + cur_classes_num
        self.cur_classes = [i + self._known_classes for i in range(cur_classes_num)]
        self.classes_per_task.append(self.cur_classes)

        # 数据包构建
        batch_size = self.args["batch_size"]
        train_indices = np.arange(self._known_classes, self._total_classes)
        test_indices = np.arange(0, self._total_classes)

        train_dataset = data_manager.get_dataset(indices=train_indices, source="train", mode="train", )
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        sa_dataset = data_manager.get_dataset(indices=train_indices, source="train", mode="strong", )
        # sa_dataset1 = data_manager.get_dataset(indices=train_indices, source="train", mode="strong", )
        # sa_dataset2 = data_manager.get_dataset(indices=train_indices, source="train", mode="strong", )
        # combined_dataset = ConcatDataset([sa_dataset, sa_dataset1, sa_dataset2])
        self.sa_loader = DataLoader(sa_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        test_dataset = data_manager.get_dataset(indices=test_indices, source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        train_dataset_for_protonet = data_manager.get_dataset(indices=train_indices, source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=batch_size, shuffle=True, num_workers=0)

        self._train()

    def _train(self,):
        self._network = self.call_model()
        if self.adapter_pool != []:
            # self._network.load_adapter(self.adapter_pool[0])
            self._network.to(self._device)
        self._network.eval()

        # 推理
        # bias = 10 * torch.randn(768)
        # feature_proto_list = [torch.randn(768) + bias for _ in self.cur_classes]
        feature_proto_list = toolkits.get_protos_with_tqdm(self.train_loader_for_protonet, self._device, self._network)
        self.previous_feature_proto_list = [] if self._known_classes == 0 else self.feature_proto_list
        self.feature_proto_list = feature_proto_list if self._known_classes == 0 else self.feature_proto_list + feature_proto_list
        self.prototypes = torch.stack(self.feature_proto_list).to(self._device)

        # 测试
        toolkits.test_accuracy(model=self._network, data_loader=self.train_loader_for_protonet,
                               prototypes=self.prototypes, epoch=-1,
                               num_epochs=0, device=self._device, words='SHOW Train')
        toolkits.test_accuracy(model=self._network, data_loader=self.test_loader,
                               prototypes=self.prototypes, epoch=-1,
                               num_epochs=0, device=self._device, words='SHOW Test')

        if self._cur_task < 10:
            self.train_vpt()
        self._network.eval()

    def train_vpt(self,):
        # 参数初始化
        self._network.train()

        # show model parameters
        # toolkits.show_model_params(self._network)

        # train meta parameters
        lrs = self.args["lr_prompt"]
        if self._cur_task > len(lrs) - 1:
            lr_ = 5.0e-3
        else:
            lr_ = lrs[self._cur_task]
        optimizer = optim.SGD(self._network.parameters(), momentum=0, lr=lr_, weight_decay=5e-4)
        loss_fn = NCMLoss()

        # 训练VPT
        train_accuracy_max = 0.0
        self.adapter_ = None
        for epoch in range(self.args["tuned_epoch"]):
            self._network.train()
            losses = 0.0

            # 创建 tqdm 进度条
            with (tqdm(total=len(self.train_loader), desc=f"Epoch [{epoch + 1}/{self.args["tuned_epoch"]}]", ncols=120) as pbar):
                for batch_counter, (_, inputs, targets) in enumerate(self.train_loader_for_protonet):
                    inputs, targets = inputs.to(self._device), targets.to(self._device)

                    # 前向传播
                    optimizer.zero_grad()
                    outputs = self._network(inputs)

                    # 准备tsne数据
                    target_long = targets.long().detach().squeeze()
                    if batch_counter == 0:
                        feature_bank, target_bank = outputs.detach().cpu(), target_long.detach().cpu()
                    else:
                        feature_bank = torch.cat((feature_bank, outputs.detach().cpu()), dim=0)
                        target_bank = torch.cat((target_bank, target_long.detach().cpu()), dim=0)

                    # loss = loss_fn.forward(features=outputs, labels=targets, prototypes=self.prototypes)
                    loss = loss_fn.modified_loss_fn_adapter(
                        current_inputs=inputs,
                        current_labels=targets,
                        prototypes=self.prototypes,
                        classes_per_task=self.classes_per_task,
                        current_task_id=self._cur_task,
                        adapter_pool=self.adapter_pool,
                        old_networks=self._old_network,
                        current_outputs=outputs,
                    )

                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()

                    # 更新进度条中的信息
                    loss_average = losses / (pbar.n + 1)
                    pbar.set_postfix(loss=loss_average)  # 显示损失和准确率
                    pbar.update(1)  # 更新进度条

            # 推理得到新的prototypes
            feature_proto_list = toolkits.get_protos_with_tqdm(self.train_loader_for_protonet, self._device, self._network)
            self.feature_proto_list = self.previous_feature_proto_list + feature_proto_list
            self.prototypes = torch.stack(self.feature_proto_list).to(self._device)

            # 测试
            train_accuracy = toolkits.test_accuracy(model=self._network, data_loader=self.train_loader_for_protonet,
                                                    prototypes=self.prototypes, epoch=epoch,
                                                    num_epochs=self.args["tuned_epoch"], device=self._device, words='Train')

            # 绘tsne图
            toolkits.tsne_classes(feature_bank, target_bank)

            # 终止epoch loop
            if train_accuracy > 97.5:   # > 97.5
                self.adapter_ = self._network.backbone.cur_adapter
                break
            if train_accuracy > train_accuracy_max:
                train_accuracy_max = train_accuracy
                self.adapter_ = self._network.backbone.cur_adapter
                # print(f'train accuracy is {train_accuracy}, save prompt {prompt_}')

        # 做Prompt的Merging
        if self.adapter_pool != []:
            alpha = 0.2
            self.adapter_ = toolkits.weighted_adapter_average(self.adapter_pool[-1], self.adapter_, alpha) # (1 - alpha) * val_old + alpha * val_adapter_
        self._network.backbone.cur_adapter = self.adapter_
        self._network.to(self._device)
        feature_proto_list = toolkits.get_protos_with_tqdm(self.train_loader_for_protonet, self._device, self._network)
        self.feature_proto_list = self.previous_feature_proto_list + feature_proto_list
        self.prototypes = torch.stack(self.feature_proto_list).to(self._device)
        toolkits.test_accuracy(model=self._network, data_loader=self.train_loader_for_protonet,
                               prototypes=self.prototypes, device=self._device, words='Merge')

        # Prototypes Drift Predict
        if 0 < self._cur_task < 10:
            self.get_prototypes_drift()
            self.feature_proto_list = self.previous_feature_proto_list + feature_proto_list
            self.prototypes = torch.stack(self.feature_proto_list).to(self._device)

        # 保存prompt_token
        self.adapter_pool.append(self.adapter_)

    # ----------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------
    def eval_accuracy(self, words='Test', top_num=1):
        model = self._network
        self._old_network.backbone.cur_adapter = self.adapter_pool[-1]
        model.to(self._device)
        data_loader = self.test_loader
        num_classes = self.prototypes.size(0)  # 总类别数
        with tqdm(total=len(data_loader), desc=f"Task{words}", ncols=120) as pbar_test:
            y_pred, y_true = [], []
            with torch.no_grad():
                for _, inputs, targets in data_loader:
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    outputs = model(inputs)

                    # 批量计算与所有原型的L2距离（高效向量化）
                    temperature = 1.0
                    distances = torch.cdist(outputs, self.prototypes, p=2)
                    logits = -distances / temperature

                    # 取Top-K预测结果
                    _, top_indices = torch.topk(logits, k=top_num, dim=1)
                    batch_preds = top_indices.cpu().tolist()  # 取Top1预测

                    y_pred.extend(batch_preds)
                    y_true.extend(targets.cpu().tolist())

                    # 更新进度条
                    test_accuracy = 100 * toolkits.top_k_accuracy(y_pred, y_true, k=1)
                    pbar_test.set_postfix(accuracy=f"{test_accuracy:.2f}%")
                    pbar_test.update(1)

        # ------------------------------------------------------------------
        # TSNE
        # ------------------------------------------------------------------
        for batch_counter, (_, inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)

            # 前向传播
            outputs = model(inputs)

            # 准备tsne数据
            target_long = targets.long().detach().squeeze()
            if batch_counter == 0:
                feature_bank, target_bank = outputs.detach().cpu(), target_long.detach().cpu()
            else:
                feature_bank = torch.cat((feature_bank, outputs.detach().cpu()), dim=0)
                target_bank = torch.cat((target_bank, target_long.detach().cpu()), dim=0)

        # 绘tsne图
        toolkits.tsne_classes(feature_bank, target_bank)

        return test_accuracy

    def get_prototypes_drift(self, ):
        self._old_network.backbone.cur_adapter = self.adapter_pool[-1]
        self._old_network.to(self._device)
        self._old_network.eval()
        self._network.eval()

        # 修改后（即时转CPU+释放显存）
        old_features_list, new_features_list = [], []
        with torch.no_grad():  # 禁用梯度计算
            for _, inputs, targets in self.sa_loader:
                inputs = inputs.to(self._device)
                # 提取后立即转CPU并释放GPU显存
                old_features = self._old_network(inputs).cpu()
                new_features = self._network(inputs).cpu()
                old_features_list.append(old_features)
                new_features_list.append(new_features)
                del inputs, old_features, new_features  # 及时删除变量

        # 以old_features_list为inputs，以new_features_list为target构造数据集
        feature_dataset = toolkits.FeatureDataset(old_features_list, new_features_list)
        dp_loader = DataLoader(feature_dataset, batch_size=32, shuffle=True)

        # train DP_network
        num_epochs = 500
        DP_network = DriftPredictNetwork(input_dim=768, hidden_dim=256).to(self._device)
        optimizer = torch.optim.SGD(DP_network.parameters(), lr=1e0)
        loss_fn = torch.nn.MSELoss()
        for epoch in range(num_epochs):
            losses = 0.0
            for inputs, targets in dp_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                optimizer.zero_grad()
                outputs = DP_network(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                losses += loss.item()

            # 记录历史损失
            if epoch % 50 == 0 or epoch > num_epochs - 4:
                print(f'Epoch{epoch+1} losses:{losses:3f}', end='; ')
            if losses < 1.5:
                break
        print()

        # 预测漂移后的prototypes
        old_prototypes = torch.stack(self.previous_feature_proto_list).to(self._device)
        new_prototypes = DP_network(old_prototypes)
        self.previous_feature_proto_list = list(new_prototypes.cpu().detach().unbind(0))


# 设计以预测prototypes漂移
class DriftPredictNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        # 编码层
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # 预测头
        self.predictor = nn.Linear(hidden_dim, input_dim)

    def forward(self, old_prototype):
        x = self.encoder(old_prototype)
        return 0.1 * self.predictor(x) + old_prototype