import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import utils.toolkits as toolkits


class BaseLeaner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        # args中参数
        # self._memory_size = args["memory_size"]
        # self._memory_per_class = args.get("memory_per_class", None)
        # self._fixed_memory = args.get("fixed_memory", False)
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def after_task(self):
        self._known_classes = self._total_classes




