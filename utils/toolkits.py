import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
import argparse
from tqdm import tqdm
from sklearn.manifold import TSNE
import os

from . import data_category
from convs.adapter import Adapter, VisionTransformer


def seed_set(seed=1):
    torch.manual_seed(seed)  # 影响CPU上随机操作
    torch.cuda.manual_seed(seed)  # 影响GPU上随机操作
    torch.cuda.manual_seed_all(seed)  # 影响多GPU上随机操作
    torch.backends.cudnn.deterministic = True  # CuDNN库以确定性模式运行,在 GPU 上相同的输入将会产生相同的输出
    torch.backends.cudnn.benchmark = False  # 禁用了CuDNN自动基准测试功能。启用时CuDNN会在多个卷积算法中选择最快的一个,可能会导致每次运行时选择不同的算法
    np.random.seed(1993)


# 加载数据集
def get_idata(dataset_name, args=None):
    dataset_name = dataset_name.lower()
    if dataset_name == "cifar100":
        return data_category.iCIFAR100()
    if dataset_name == "cifar224":
        return data_category.iCIFAR224()
    if dataset_name == "cub":
        return data_category.CUB()
    if dataset_name == "imagenet-a":
        return data_category.iImageNetA()
    if dataset_name == "imagenet-r":
        return data_category.iImageNetR()
    if dataset_name == "vtab":
        return data_category.vtab()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


# 创建split_dataset编队_increments
def get_increments_list(init_cls, increment, _class_order):
    assert init_cls <= len(_class_order), "No enough classes."
    _increments = [init_cls]
    while sum(_increments) + increment < len(_class_order):
        _increments.append(increment)
    offset = len(_class_order) - sum(_increments)
    if offset > 0:
        _increments.append(offset)
    # print(_increments)  # cifar100_inc5: [5,5,...,5]
    return _increments


def get_model(model_name, args):
    name = model_name.lower()
    if name == 'simplecil':
        from models0.simplecil import Learner
        print('--model simplecil')
        return Learner(args)
    elif name == 'constantcoordinatecil':
        from models0.constantcoordinatecil import Learner
        print('--model constantcoordinatecil')
        return Learner(args)
    elif name == 'crazyprotocil':
        from models0.crazyprotocil import Learner
        print('--model crazyprotocil')
        return Learner(args)
    elif name == 'gdprotocil':
        from models0.gdprotocil import Learner
        print('--model gdprotocil')
        return Learner(args)
    elif name == 'kmeanscil':
        from models0.kmeanscil import Learner
        print('--model kmeanscil')
        return Learner(args)
    elif name == 'gdprotoscil':
        from models0.gdprotoscil import Learner
        print('--model gdprotoscil')
        return Learner(args)
    elif name == 'apervpt':
        from models0.apervpt import Learner
        print('--model apervpt')
        return Learner(args)
    elif name == 'apervpt_simplecil':
        from models0.apervpt_simplecil import Learner
        print('--model apervpt_simplecil')
        return Learner(args)
    elif name == 'contrastcil':
        from models1.contrastcil import Learner
        print('--model contrastcil')
        return Learner(args)
    elif name == 'ncmlosscil':
        from models1.ncmlosscil import Learner
        print('--model ncmlosscil')
        return Learner(args)
    elif name == 'mine11':
        from models1.mine11cil import Learner
        print('--model mine11')
        return Learner(args)
    elif name == 'ncmlosscil_ir':
        from models1.ncmlosscil_ir import Learner
        print('--model ncmlosscil_ir')
        return Learner(args)
    else:
        raise NotImplementedError("Unknown model {}.".format(model_name))


def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Beginning Your Incremental Class Learning Experiment.')
    parser.add_argument('--config', type=str, default='./exps/cifar_in21k.json', help='Json file of settings.')
    return parser


def show_model_params(network):
    # Freeze the parameters for ViT.
    total_params = sum(p.numel() for p in network.parameters())
    print(f'{total_params:,} total parameters.', end='\t')
    total_trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.', end='\t')

    # if some parameters are trainable, print the key name and corresponding parameter number
    if total_params != total_trainable_params:
        for name, param in network.named_parameters():
            if param.requires_grad:
                print(name, param.numel(), end='\t')
    print()

def classify_with_proto(test_output, prototypes, top_num=2):
    """
    基于原型（L2距离）的分类函数
    Args:
        test_output: 测试样本特征 [768]
        prototypes: 所有类别原型 [35, 768]
        top_num: 返回前top_num个预测结果
    Returns:
        predict: 预测的类别索引列表（长度=top_num）
    """
    # 确保数据在CPU（若prototypes在GPU）
    test_output = test_output.cpu()
    prototypes = prototypes.cpu()

    # 向量化计算所有类别的L2距离（形状 [35]）
    distances = torch.sum((test_output - prototypes) ** 2, dim=1)  # [35]

    # 计算相似度（取距离倒数，避免除零）
    epsilon = 1e-8  # 防止距离为0导致无穷大
    similarities = 1 / (distances + epsilon)  # [35]

    # 直接取Top-K类别索引
    top_sim, top_indices = torch.topk(similarities, k=top_num, largest=True, sorted=True)
    predict = top_indices.tolist()

    return predict[:top_num]  # 确保返回长度一致


def top_k_accuracy(y_pred, y_true, k):
    correct_count = 0
    total_count = len(y_true)

    for i in range(total_count):
        # 检查真实标签是否在预测的前 k 个类别中
        if y_true[i] in y_pred[i][:k]:
            correct_count += 1

    # 计算 top-k 准确率
    accuracy = correct_count / total_count
    return accuracy


def get_protos_with_tqdm(data_loader, device, model):
    model.to(device)
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for _, inputs, targets in tqdm(data_loader, desc="Inference", ncols=120):
            inputs = inputs.to(device)
            embedding = model(inputs)
            embedding_list.append(embedding.cpu())
            label_list.append(targets.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    # NCM, 对class’s features取mean
    class_list = np.unique(label_list)
    feature_proto_list = []
    for class_index in class_list:
        data_index = (label_list == class_index).nonzero().squeeze(-1)
        embeddings = embedding_list[data_index]
        proto = embeddings.mean(0)
        feature_proto_list.append(proto)

    return feature_proto_list


def get_protos(data_loader, device, model):
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            embedding = model.forward_features_(inputs)
            embedding_list.append(embedding.cpu())
            label_list.append(targets.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    # NCM, 对class’s features取mean
    class_list = np.unique(label_list)
    feature_proto_list = []
    for class_index in class_list:
        data_index = (label_list == class_index).nonzero().squeeze(-1)
        embeddings = embedding_list[data_index]
        proto = embeddings.mean(0)
        feature_proto_list.append(proto)

    return feature_proto_list


def test_accuracy(model, data_loader, prototypes, epoch=-1, num_epochs=0, device='cuda', words='Test', top_num=2):
    model.eval()
    with tqdm(total=len(data_loader), desc=f"{words} Epoch [{epoch + 1}/{num_epochs}]",
              ncols=120) as pbar_test:
        y_pred, y_true = [], []
        with torch.no_grad():  # 不计算梯度
            for _, inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                # 批量计算与所有原型的L2距离（高效向量化）
                temperature = 1.0
                distances = torch.cdist(outputs, prototypes, p=2)
                logits = -distances / temperature

                # 取Top-K预测结果
                _, top_indices = torch.topk(logits, k=top_num, dim=1)
                batch_preds = top_indices.cpu().tolist()  # 取Top1预测

                y_pred.extend(batch_preds)
                y_true.extend(targets.cpu().tolist())

                # 更新测试进度条信息
                test_accuracy = 100 * top_k_accuracy(y_pred=y_pred, y_true=y_true, k=1)
                pbar_test.set_postfix(accuracy=f"{test_accuracy:.2f}%")
                pbar_test.update(1)

    return test_accuracy


def tsne_classes(feature_bank, target_bank):
    # 假设 feature_bank 和 target_bank 已经转换为 NumPy 数组
    feature_bank = feature_bank.numpy()
    target_bank = target_bank.numpy()

    # 执行 t-SNE 降维

    os.environ["LOKY_MAX_CPU_COUNT"] = "10"  # 假设你有4个CPU核心
    tsne = TSNE(n_components=2, random_state=0, n_jobs=1)
    output = tsne.fit_transform(feature_bank)

    # 获取唯一的类别标签
    unique_labels = np.unique(target_bank)

    # 示例：自定义 12 种高对比度颜色（HEX 格式）
    custom_colors = [
        '#1f77b4', '#ff7f0e', '#0000FF', '#FF00FF',
        '#9467bd', '#FFD700', '#00FFFF', '#7f7f7f',
        '#FF1493', '#17becf', '#4B0082', '#FF4500'
    ]

    # 绘制散点图
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(unique_labels):
        index = (target_bank == label)
        color = custom_colors[i % len(custom_colors)]  # 循环使用颜色
        plt.scatter(
            output[index, 0],
            output[index, 1],
            s=40,
            color=color,
            edgecolors='k',
            linewidths=0.3,
            alpha=0.8,
            label=f'Class {label}'
        )

    plt.legend(markerscale=2)
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()


def weighted_prompt_average(_prompt: dict, prompt_: dict, alpha: float, beta=0.5) -> dict:
    new_prompt = type(prompt_)()  # 保持原有结构（如OrderedDict）

    # 确保键完全一致
    assert _prompt.keys() == prompt_.keys(), "Prompt字典的层结构不匹配"

    for key in prompt_.keys():
        val_prompt = prompt_[key]
        val_old = _prompt[key]

        if np.random.uniform(0, 1) < beta and alpha > 0.0:
            # 递归处理嵌套的OrderedDict（如多层级结构）
            if isinstance(val_prompt, dict):
                new_prompt[key] = weighted_prompt_average(val_old, val_prompt, alpha, beta)
            else:
                # 对Tensor进行加权计算
                new_prompt[key] = (1 - alpha) * val_old + alpha * val_prompt
        else:
            new_prompt[key] = val_old

    return new_prompt


def weighted_adapter_average(old_adapter, new_adapter, alpha):
    # 创建新 Adapter 实例（继承原配置）
    vit = VisionTransformer()
    merged_adapter = vit.cur_adapter

    # 合并参数
    merged_state_dict = {}
    for key in old_adapter.state_dict():
        old_val = old_adapter.state_dict()[key]
        new_val = new_adapter.state_dict()[key]
        merged_state_dict[key] = (1 - alpha) * old_val + alpha * new_val

    # 加载合并后的参数到新实例（使用 assign=True 保持参数属性）
    merged_adapter.load_state_dict(merged_state_dict, assign=True)

    return merged_adapter


def plot_class_similarities(cosine_similarity_list):
    plt.figure(figsize=(12, 6))

    # 提取类别数量和任务数量
    num_classes = len(cosine_similarity_list[0])  # 从第一个任务获取类别数
    num_tasks = len(cosine_similarity_list)

    # 创建颜色映射
    colors = [
        '#1f77b4', '#ff7f0e', '#0000FF', '#FF00FF',
        '#9467bd', '#FFD700', '#00FFFF', '#7f7f7f',
        '#FF1493', '#17becf', '#4B0082', '#FF4500'
    ]

    # 为每个类别绘制折线
    for class_idx in range(num_classes):
        # 提取当前类别在所有任务中的相似度值
        y = [tensor[class_idx].item() if isinstance(tensor, (list, tuple)) else tensor.item()
             for tensor in cosine_similarity_list]

        # 生成x轴坐标（任务索引）
        x = list(range(num_tasks))

        plt.plot(x, y,
                 marker='o',
                 linestyle='-',
                 color=colors[class_idx],
                 label=f'Class {class_idx}',
                 alpha=0.8,
                 linewidth=2,
                 markersize=8)
    # 美化图表
    plt.title("Cosine Similarity Evolution Across Tasks", fontsize=14, pad=20)
    plt.xlabel("Task Index", fontsize=12, labelpad=10)
    plt.ylabel("Cosine Similarity", fontsize=12, labelpad=10)
    plt.xticks(range(num_tasks), [f"Task {i}" for i in range(num_tasks)])
    plt.ylim(0.9, 1.01)  # 根据示例数据调整Y轴范围
    plt.grid(True, linestyle='--', alpha=0.6)

    # 添加图例（自动适应类别数量）
    plt.legend(bbox_to_anchor=(1.05, 1),
               loc='upper left',
               borderaxespad=0.,
               fontsize=10,
               ncol=1 if num_classes <= 5 else 2)  # 类别多时分成两列
    plt.show()


class FeatureDataset(Dataset):
    def __init__(self, old_features_list, new_features_list, transform=None):
        """
        将新旧特征列表转换为PyTorch Dataset
        Args:
            old_features_list (list): 旧模型特征列表，每个元素为(batch_size, feature_dim)张量
            new_features_list (list): 新模型特征列表，每个元素形状与old_features_list对应
            transform (callable, optional): 可选的同步特征变换函数
        """
        # 拼接所有批次的特征张量 (总_samples, feature_dim)
        self.old_features = torch.cat(old_features_list, dim=0)
        self.new_features = torch.cat(new_features_list, dim=0)
        self.transform = transform

        # 维度校验 (确保新旧特征对齐)
        assert len(self.old_features) == len(self.new_features), "特征数量不匹配"
        assert self.old_features.shape[1] == self.new_features.shape[1], "特征维度不匹配"

    def __len__(self):
        return self.old_features.size(0)

    def __getitem__(self, idx):
        old_feat = self.old_features[idx]
        new_feat = self.new_features[idx]

        if self.transform:
            old_feat, new_feat = self.transform(old_feat, new_feat)

        return old_feat, new_feat