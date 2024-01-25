import random
import numpy as np
import torch

"""
设置随机数种子,确保了程序中使用的各个随机数生成器都受到相同的初始种子的影响，
从而使得程序的随机性可控，能够在不同运行中产生相同的随机序列。
这对于实验的可重复性、调试和结果的一致性都是有帮助的。
"""


def setup_seed(seed):
    np.random.seed(seed)  # 设置NumPy库的随机种子，影响NumPy库中的随机数生成
    random.seed(seed)  # 设置Python内建的random模块的随机种子
    torch.manual_seed(seed)  # 设置PyTorch的CPU随机种子
    torch.cuda.manual_seed(seed)  # 设置PyTorch的当前GPU的随机种子
    torch.cuda.manual_seed_all(seed)  # 设置PyTorch所有GPU的随机种子
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = True
