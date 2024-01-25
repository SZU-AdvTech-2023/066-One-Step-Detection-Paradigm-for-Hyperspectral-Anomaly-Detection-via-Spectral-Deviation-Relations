import hydra
import sys
import os
from omegaconf import DictConfig
from utils.logging_ import logger_config
from importlib import import_module
from pathlib import Path
import logging
import torch
from utils.random_seed import setup_seed
import warnings
import torch.multiprocessing as mp

warnings.filterwarnings('ignore')

if len(sys.argv) > 1:  # 关键的一步
    config_path = sys.argv[1]
    sys.argv.pop(1)


# config_path = './configs/HYDICE/TDD_config.yaml'

# 配置文件的地址
@hydra.main(config_path="D:\PapersCode\TDD-master-zt\configs\ABU", config_name="TDD_config")
def main(cfg: DictConfig) -> None:  # 参数cfg的类型是DictConfig。他通常是OmegaConf库的一部分，用于处理配置信息的数据结构。
    setup_seed(cfg.random_seed)  # 设置随机数种子 utils/random_seed.py
    torch.autograd.set_detect_anomaly(True)  # 启用了PyTorch中的异常检测。当启用异常检测时，PyTorch会在计算图中发现潜在错误或不稳定性时引发异常，以帮助调试代码。

    working_dir = str(Path.cwd())
    logging.info(f"The current working directory is {working_dir}")  # 打印当前工作目录的信息

    runner_module_cfg = cfg.runner_module
    module_path, attr_name = runner_module_cfg.split(" - ")
    module = import_module(module_path)
    runner_module = getattr(module, attr_name)
    runner = runner_module(cfg, working_dir)

    runner.run()


if __name__ == "__main__":
    main()
