import hydra
from runners.base_runner import BaseRunner
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple
from numpy import ndarray as NDArray
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
from albumentations.pytorch import ToTensorV2
from utils.savefig import save_heatmap
from utils.average_meter import AverageMeter
import time
from utils.overlap_infer import overlap_infer 
from metrics import compute_auroc
import albumentations as A
from utils.img_io import read_img, write_img
import os
from tqdm import tqdm 
import time



class runner_TDD(BaseRunner):
    # 训练深度学习模型
    def _train(self, epoch: int) -> None:
        self.model.train()# 将模型设置为训练模式
        train_iter_loss = AverageMeter()# 损失平均值
        epoch_start_time = time.time()
        self.train_loader_size = self.dataloaders['train'].__len__()
        # 优化器
        opt_en = torch.optim.Adam(self.model.encoder.parameters(), lr=0.001, weight_decay=0.0005)
        opt_dec = torch.optim.Adam(self.model.decoder.parameters(), lr=0.001 * 10, weight_decay=0.0005)

        for batch_idx, (img, mask) in enumerate(self.dataloaders["train"]):

            # 将优化器的梯度缓存清零
            opt_en.zero_grad()
            opt_dec.zero_grad()

            # 将输入数据移到指定的计算设备，通常是GPU
            img = img.to(self.cfg.params.device)
            mask = mask.to(self.cfg.params.device)

            # 通过模型进行前向传播，得到预测结果pred和损失值loss
            pred, loss = self.model(img, mask.unsqueeze(1))

            loss.backward()  # 计算损失相对于模型参数的梯度
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)  # 对模型的梯度进行裁剪，以防止梯度爆炸

            # 分别对 decoder 和 encoder 的参数进行优化，更新参数
            opt_dec.step()
            opt_en.step()

            # 更新累积的训练损失
            train_iter_loss.update(loss.item())

            if batch_idx % 5 == 0: 
                spend_time = time.time() - epoch_start_time
                logging.info('[train] epoch:{} iter:{}/{} {:.2f}% lr:{:.6f} loss:{:.6f} ETA:{}min'.format(epoch, batch_idx, self.train_loader_size, 
                batch_idx / self.train_loader_size * 100, self.optimizer.param_groups[-1]['lr'], train_iter_loss.avg, spend_time / (batch_idx + 1) * self.train_loader_size // 60 - spend_time // 60))
                train_iter_loss.reset()  # 重置用于累积训练损失的对象，以便记录下一个 batch 的损失


    # 测试（推理）深度学习模型
    def _test(self, epoch: int) -> None:
        self.model.eval() # 将模型切换到评估模式
        save_dir = os.path.join(self.working_dir, "epochs-" + str(epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not hasattr(self, 'test_images') or not hasattr(self, 'test_img_gts'):
            self.test_images = []
            self.test_img_gts = []
            self.test_file_names = []
            self.test_patch_sizes = []
            self.test_pad_sizes = []
            self.test_input_sizes = []

            for test_image, test_img_gt, test_file_name, test_patch_size, test_pad_size, test_input_size in (self.dataloaders["test"]):
                self.test_images.append(test_image)
                self.test_img_gts.append(test_img_gt)
                self.test_file_names.append(test_file_name[0])
                self.test_patch_sizes.append(test_patch_size[0])
                self.test_pad_sizes.append(test_pad_size[0])
                self.test_input_sizes.append(test_input_size[0])
            
        for i in tqdm(range(len(self.test_images))):

            mb_img = self.test_images[i]
            mb_gt = self.test_img_gts[i]
            file_name = self.test_file_names[i]
            file_name = str(i) + '_' + file_name

            artifacts: Dict[str, List[NDArray]] = {
                "img": [],
                "gt": [],
                "amap": [],  # 异常图
            }

            logging.info("test model on the image: " + file_name)

            with torch.no_grad():
                if mb_img.shape[1] > self.cfg.params.training_channels:
                    slice_number = int(mb_img.shape[1]/self.cfg.params.training_channels + 1)
                else:
                    slice_number = 1
                predicted_test_overlap = 0
                for j in range(slice_number):
                    logging.info("test on the " + str(j) + " slice")
                    if j == slice_number - 1:
                        img_t = mb_img[:,-self.cfg.params.training_channels:,:,:]
                    else:
                        img_t = mb_img[:,j:j+self.cfg.params.training_channels,:,:]
                    cfg_test = dict(title_size=[self.test_patch_sizes[i], self.test_patch_sizes[i]],
                    pad_size=[self.test_pad_sizes[i], self.test_pad_sizes[i]], batch_size=1,  
                    padding_mode='mirror', num_classes=1, device=self.cfg.params.device, test_size=self.test_input_sizes[i])  
                    data_overlap_input = img_t.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
                    predicted_test_overlap = (predicted_test_overlap + overlap_infer(cfg_test, model=self.model, img=data_overlap_input)['score_map'])/2.0

                artifacts["amap"].extend(predicted_test_overlap.transpose(2,0,1))

            # 将测试图像、ground truth和异常图存储到artifacts字典中
            artifacts["img"].extend(mb_img.permute(0, 2, 3, 1).detach().cpu().numpy()) 
            artifacts["gt"].extend((1-mb_gt).detach().cpu().numpy())
            artifacts["amap"] = np.array(artifacts["amap"]) 

            # 异常图归一化
            artifacts["amap"] = (artifacts["amap"] - artifacts["amap"].min())/(artifacts["amap"].max() - artifacts["amap"].min())

            save_path = os.path.join(self.working_dir, "epochs-" + str(epoch))
            if not os.path.exists(save_path):
                os.makedirs(save_path) 

            anomaly_map = artifacts["amap"][0, :, :] 
            save_ratio = 20

            write_img((anomaly_map), os.path.join(save_path, file_name + '_anomaly_map.tif'))
            save_heatmap(anomaly_map, save_path, save_height=anomaly_map.shape[0]/save_ratio, save_width=anomaly_map.shape[1]/save_ratio ,dpi=150, file_name=file_name) 
            
            try:
                auroc = compute_auroc(epoch, np.array(artifacts["amap"]), np.array(artifacts["gt"]), self.working_dir)
            except IndexError:
                logging.info('Error happened when computing AUC')
                pass

            self.save(epoch)

    def save(self, epoch):
        save_path = os.path.join(self.working_dir, "epochs-" + str(epoch))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = os.path.join(save_path, 'checkpoint-model.pth')
        torch.save(self.model.state_dict(), filename)

    def _re_init_optimizer(self, params) -> Optimizer:
        cfg = self.cfg.optimizer
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), params=params)

    def _re_init_scheduler(self, optimizer) -> _LRScheduler:

        cfg = self.cfg.scheduler
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), optimizer=optimizer)


    def load(self, save_dir):
        filename = os.path.join(save_dir, 'checkpoint-model.pth')
        self.model.load_state_dict(torch.load(filename, map_location=torch.device(self.cfg.params.device)))