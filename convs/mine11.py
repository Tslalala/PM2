import copy
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from convs.adapter import vit_base_patch16_224_in21k


class Mine11(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_dim = 768
        self.use_init_ptm = False
        self._device = 'cuda'
        self.backbone = vit_base_patch16_224_in21k()

    def freeze(self):
        for _, param in self.named_parameters():
            param.requires_grad = False

    def copy(self):
        return copy.deepcopy(self)

    def extract_prototype(self, dataset, data_loader, adapter):
        with torch.no_grad():
            prog_bar = tqdm(data_loader)

            # extract embeddings
            embedding_list, label_list = [], []
            for _, batch in enumerate(prog_bar):
                (_, data, label) = batch
                data = data.to(self.device)
                label = label.to(self.device)
                embedding = self.backbone.forward_proto(data, adapter)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

                prog_bar.set_description('Inference:')

            embedding_list = torch.cat(embedding_list, dim=0)
            label_list = torch.cat(label_list, dim=0)

            # construct prototype-based classifier
            class_list = np.unique(dataset.labels)
            proto_list = []
            for class_index in class_list:
                data_index = (label_list == class_index).nonzero().squeeze(-1)
                embedding = embedding_list[data_index]
                proto = embedding.mean(0)
                proto_list.append(proto[None])

            return torch.cat(proto_list, dim=0)

    def forward(self, x):
        x = self.backbone.forward_train(x)
        return x