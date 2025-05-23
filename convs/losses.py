import torch
import torch.nn as nn
import torch.nn.functional as F
from convs.vpt import build_promptmodel


class PPLoss(nn.Module):
    def __init__(self, delta=10.0, alpha=1.0, beta=0.1, reduction='mean'):
        """
        Args:
            delta (float): Push Loss的间隔阈值，类间原型距离需大于delta，否则惩罚。
            alpha (float): Pull Loss的权重系数（类内聚合强度）。
            beta (float): Push Loss的权重系数（类间分离强度）。
            reduction (str): 损失计算方式，可选 'mean' 或 'sum'。
        """
        super().__init__()
        self.unique_labels = None
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(self, features, labels, prototypes=None):
        """
        Args:
            features (Tensor): 输入特征向量，形状 [B, D]
            labels (Tensor): 类别标签，形状 [B]
            prototypes (Tensor, optional): 外部传入的类别原型矩阵，形状 [C, D]

        Returns:
            total_loss (Tensor): 总损失（Pull + Push）
            pull_loss (Tensor): 类内聚合损失
            push_loss (Tensor): 类间分离损失
        """
        # 如果未传入外部原型，则动态计算当前batch的类别原型
        if prototypes is None:
            prototypes = self._compute_prototypes(features, labels)

        # 处理labels
        unique_labels = torch.unique(labels)
        self.unique_labels = unique_labels

        # 计算Pull Loss：类内聚合（同类特征靠近原型）
        pull_loss = self._compute_pull_loss(features, labels, prototypes)

        # 计算Push Loss：类间分离（不同类原型间距大于delta）
        push_loss = self._compute_push_loss(prototypes)

        # 总损失加权和
        total_loss = self.alpha * pull_loss + self.beta * push_loss

        return total_loss

    def _compute_prototypes(self, features, labels):
        """
        动态计算当前batch中每个类别的原型（均值）
        形状：
            features: [B, D]
            labels: [B]
        Returns:
            prototypes: [C_batch, D]，C_batch为当前batch中实际存在的类别数
        """
        prototypes = []
        for c in self.unique_labels:
            mask = (labels == c)
            class_features = features[mask]
            proto = class_features.mean(dim=0)  # [D]
            prototypes.append(proto)
        return torch.stack(prototypes, dim=0)  # [C_batch, D]

    def _compute_pull_loss(self, features, labels, prototypes):
        """
        计算类内聚合损失：所有样本特征与对应类别原型的平均距离（固定35类）
        Args:
            features: [B, 768]
            labels: [B]（标签范围0~34）
            prototypes: [35, 768]
        """
        # 根据标签索引对应的原型 [B, 768]
        target_prototypes = prototypes[labels]  # 直接通过labels索引

        # 计算每个样本特征与原型的平方距离 [B]
        distances = torch.sum((features - target_prototypes) ** 2, dim=1)

        # 平均距离作为Pull Loss
        pull_loss = torch.mean(distances)
        return pull_loss

    def _compute_push_loss(self, prototypes):
        """
        计算类间分离损失：强制35个类原型间距大于delta
        Args:
            prototypes: [35, 768]
        """
        # 计算类别数
        num_classes = self.unique_labels.size(0)

        # 计算所有原型对的欧氏距离 [num_classes, num_classes]
        pairwise_dist = torch.cdist(prototypes, prototypes, p=2)

        # 排除对角线（自身距离）
        mask = 1 - torch.eye(num_classes, device=prototypes.device)
        penalty = torch.relu(self.delta - pairwise_dist * mask)  # [num_classes, num_classes]

        # 仅计算上三角部分（避免重复计算）
        upper_tri = torch.triu_indices(num_classes, num_classes, offset=1)
        push_loss = torch.mean(penalty[upper_tri[0], upper_tri[1]] ** 2)

        return push_loss

    def extra_repr(self):
        return f"delta={self.delta}, alpha={self.alpha}, beta={self.beta}"


class NCMLoss(nn.Module):
    def __init__(self, temperature=1.0, epsilon=1e-8):
        """
        Args:
            temperature (float): 温度系数，用于缩放距离影响（类似对比学习）
            epsilon (float): 数值稳定项，防止除零错误
        """
        super().__init__()
        self.temperature = temperature
        self.epsilon = epsilon

    def forward(self, features, labels, prototypes=None):
        # 动态计算原型（若未提供）
        if prototypes is None:
            raise ValueError("prototypes is None")

        # 计算特征与所有原型的距离 [B, C]
        distances = torch.cdist(features, prototypes, p=2)  # 欧氏距离

        # 将距离转换为概率（距离越小概率越高）
        logits = - distances / self.temperature  # [B, C]
        loss = F.cross_entropy(logits, labels)

        # 新增对比正则化项
        C = prototypes.size(0)
        if C > 1:
            # 生成标签对：所有不同类对组合
            label_pairs = torch.combinations(torch.arange(C, device=prototypes.device), r=2)
            anchor = prototypes[label_pairs[:, 0]]  # [num_pairs, D]
            positive = prototypes[label_pairs[:, 1]]  # 实际为负样本（不同类）

            # 计算对比损失（增大负样本间距）
            margin = 10.0  # 目标最小间距
            contrastive_loss = F.relu(margin - torch.norm(anchor - positive, dim=1)).mean()
        else:
            contrastive_loss = 0.0

        combined_loss = loss + 0.2 * contrastive_loss
        return combined_loss

    def extra_repr(self):
        return f"temperature={self.temperature}, epsilon={self.epsilon}"

    def modified_loss_fn(self,
                         current_features,
                         old_features_list,
                         labels,
                         prototypes,
                         current_loss_weight=1.0,
                         old_consistency_weight=0.12
                         ):
        """
        改进的损失函数，整合当前任务损失和旧模型一致性约束

        Args:
            current_features (Tensor): 当前模型输出特征 [B, D]
            old_features_list (List[Tensor]): 历史模型输出特征列表 [n_old_models, B, D]
            labels (Tensor): 当前批次标签 [B]
            prototypes (Tensor): 所有类原型 [C, D]
            current_loss_weight (float): 当前分类损失权重
            old_consistency_weight (float): 旧模型一致性约束权重
            temperature (float): 相似度分布温度系数

        Returns:
            total_loss (Tensor): 整合后的总损失
        """
        # 原始分类损失（当前模型）
        distances = torch.cdist(current_features, prototypes, p=2)  # [B, C]
        logits = - distances / self.temperature
        loss_current = F.cross_entropy(logits, labels) * current_loss_weight

        # 旧模型一致性约束项（鼓励与旧模型输出差异）
        consistency_loss = 0
        for old_features in old_features_list:
            # 计算特征相似度（鼓励差异）
            similarity = F.cosine_similarity(current_features, old_features, dim=1)  # [B]
            consistency_loss += similarity.mean()  # 最大化差异即最小化相似度

        # 总损失 = 分类损失 - 相似度损失（因为要最小化总损失，等价于最大化差异）
        total_loss = loss_current - old_consistency_weight * consistency_loss

        return total_loss


    def modified_loss_fn_(
            self,
            current_inputs,
            current_labels,
            prototypes,
            classes_per_task,
            current_task_id,
            prompt_pool,
            old_networks,
            current_outputs,
    ):
        device = prototypes.device
        batch_size = current_inputs.size(0)
        num_classes = prototypes.size(0)

        # ==================== 构建累积相似度矩阵 ====================
        # 构造累积相似度矩阵
        total_similarities = torch.zeros(batch_size, num_classes).to(device)

        # 遍历所有任务
        for task_id, prompt in enumerate(prompt_pool):
            # 加载对应任务的prompt参数
            old_networks.load_prompt(prompt)
            old_networks.to(device)

            # 获取该任务的类索引范围
            task_class_indices = classes_per_task[task_id]
            start_idx = min(task_class_indices)
            end_idx = max(task_class_indices) + 1  # 包含末端索引

            # 提取该任务对应的原型 [n_classes_in_task, D]
            task_prototypes = prototypes[start_idx:end_idx]

            # 计算相似度（与eval_accuracy完全一致）
            outputs = old_networks(current_inputs)  # [B, D]
            distances = torch.cdist(outputs, task_prototypes, p=2)  # [B, num_task_classes]
            similarities = - distances / self.temperature

            # 累加到总相似度矩阵的对应位置
            total_similarities[:, start_idx:end_idx] += similarities

        # ========== 新增当前任务相似度计算 ==========
        # 获取当前任务类索引范围
        current_class_indices = classes_per_task[current_task_id]
        start_idx = min(current_class_indices)
        end_idx = max(current_class_indices) + 1

        # 提取当前任务原型
        current_prototypes = prototypes[start_idx:end_idx]

        # 计算当前任务相似度（不加载额外Prompt）
        current_distances = torch.cdist(current_outputs, current_prototypes, p=2)
        current_similarities = - current_distances / self.temperature

        # 累加到总相似度矩阵
        total_similarities[:, start_idx:end_idx] += current_similarities

        # ==================== 计算交叉熵损失 ====================
        # 直接使用全局标签（无需重映射）
        ce_loss = F.cross_entropy(total_similarities, current_labels)

        # ==================== 原型间距正则项 ====================
        # 计算所有原型对之间的L2距离
        proto_dist = torch.cdist(prototypes, prototypes, p=2)  # [C, C]

        # 掩码排除自身距离（对角线）
        mask = ~torch.eye(prototypes.size(0), dtype=torch.bool, device=prototypes.device)
        valid_dist = proto_dist[mask].view(prototypes.size(0), -1)  # [C, C-1]

        # 计算每个原型到最近邻的距离
        min_distances, _ = valid_dist.min(dim=1)  # [C]

        # 鼓励最近邻距离大于margin（Hinge Loss形式）
        margin = 2.0
        spread_loss = F.relu(margin - min_distances).mean()

        # 正则项权重（建议0.1-0.3）
        lambda_spread = 0.2
        p_loss = spread_loss * lambda_spread

        # 总损失
        total_loss = ce_loss + p_loss

        return total_loss

    def modified_loss_fn_adapter(
            self,
            current_inputs,
            current_labels,
            prototypes,
            classes_per_task,
            current_task_id,
            adapter_pool,
            old_networks,
            current_outputs,
    ):
        device = prototypes.device
        batch_size = current_inputs.size(0)
        num_classes = prototypes.size(0)

        # ==================== 构建累积相似度矩阵 ====================
        # 构造累积相似度矩阵
        total_similarities = torch.zeros(batch_size, num_classes).to(device)

        # 遍历所有任务
        for task_id, adapter in enumerate(adapter_pool):
            # 加载对应任务的prompt参数
            # old_networks.load_prompt(adapter)
            old_networks.backbone.cur_adapter = adapter
            old_networks.to(device)

            # 获取该任务的类索引范围
            task_class_indices = classes_per_task[task_id]
            start_idx = min(task_class_indices)
            end_idx = max(task_class_indices) + 1  # 包含末端索引

            # 提取该任务对应的原型 [n_classes_in_task, D]
            task_prototypes = prototypes[start_idx:end_idx]

            # 计算相似度（与eval_accuracy完全一致）
            outputs = old_networks(current_inputs)  # [B, D]
            distances = torch.cdist(outputs, task_prototypes, p=2)  # [B, num_task_classes]
            similarities = - distances / self.temperature

            # 累加到总相似度矩阵的对应位置
            total_similarities[:, start_idx:end_idx] += similarities

        # ========== 新增当前任务相似度计算 ==========
        # 获取当前任务类索引范围
        current_class_indices = classes_per_task[current_task_id]
        start_idx = min(current_class_indices)
        end_idx = max(current_class_indices) + 1

        # 提取当前任务原型
        current_prototypes = prototypes[start_idx:end_idx]

        # 计算当前任务相似度（不加载额外Prompt）
        current_distances = torch.cdist(current_outputs, current_prototypes, p=2)
        current_similarities = - current_distances / self.temperature

        # 累加到总相似度矩阵
        total_similarities[:, start_idx:end_idx] += current_similarities

        # ==================== 计算交叉熵损失 ====================
        # 直接使用全局标签（无需重映射）
        ce_loss = F.cross_entropy(total_similarities, current_labels)

        # ==================== 原型间距正则项 ====================
        # 计算所有原型对之间的L2距离
        proto_dist = torch.cdist(prototypes, prototypes, p=2)  # [C, C]

        # 掩码排除自身距离（对角线）
        mask = ~torch.eye(prototypes.size(0), dtype=torch.bool, device=prototypes.device)
        valid_dist = proto_dist[mask].view(prototypes.size(0), -1)  # [C, C-1]

        # 计算每个原型到最近邻的距离
        min_distances, _ = valid_dist.min(dim=1)  # [C]

        # 鼓励最近邻距离大于margin（Hinge Loss形式）
        margin = 2.0
        spread_loss = F.relu(margin - min_distances).mean()

        # 正则项权重（建议0.1-0.3）
        lambda_spread = 0.2
        p_loss = spread_loss * lambda_spread

        # 总损失
        total_loss = ce_loss + p_loss

        return total_loss