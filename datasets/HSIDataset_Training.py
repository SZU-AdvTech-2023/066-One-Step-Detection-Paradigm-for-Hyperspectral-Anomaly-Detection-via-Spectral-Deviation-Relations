import logging
import os
import sys

sys.path.append("./../")
import torch
from skimage import io
from torch.utils.data import Dataset
from utils.img_io import read_img, write_img
import numpy as np
from albumentations.pytorch import ToTensorV2
from random import choice
import albumentations as A
import random
from skimage import transform as sk_transform

# 加载和预处理用于训练神经网络模型的超光谱图像

class HSIDataset_Training(Dataset): 
    def __init__(self, img_dir, max_height, min_height, transforms):
        self.img_dir = img_dir
        self.transform = transforms

        self.img_files = os.listdir(self.img_dir)

        self.dataset_length = len(os.listdir(self.img_dir)) 
        self.location_selection = A.Compose([     
                A.CoarseDropout(max_holes=20 , min_holes=10, max_height=max_height, max_width=max_height, min_height=min_height, min_width=min_height, fill_value=1, mask_fill_value=1, p=1.0)
            ])
        # 初始化location_selection、shift_transform和shuffle_transform属性，用于对超光谱图像进行随机遮挡、平移、缩放、旋转和通道打乱。
        self.shift_transform =A.Compose([A.ShiftScaleRotate(shift_limit=0.0, p=1), A.IAAPiecewiseAffine(scale=(0.1, 0.3), p=1.0, order=0)])
        self.shuffle_transform = A.ChannelShuffle(p=1.0)

        logging.info(f'Creating dataset with {self.dataset_length} examples') 

    def __len__(self): 
        return self.dataset_length 

    def __getitem__(self, i):
 
        file_name = self.img_files[i] 
             
        img_path = os.path.join(self.img_dir, file_name)

        img = read_img(img_path=img_path)

        # 过程1
        mask = np.zeros((img.shape[0], img.shape[1]))   # 创建一个全零的掩码。

        sample = self.location_selection(image=img, mask = mask)  # 变换对图像和掩码进行随机遮挡。

        # 过程2
        p_t = random.random() 
        img2 = self.shuffle_transform(image=img)['image']  # 使用 shuffle_transform 变换对X每个像素在光谱维度上进行随机洗牌
        
        locs = np.where(sample['mask'] == 1)
        sample['image'][locs[0], locs[1], :] = img2[locs[0], locs[1], :]  # 将遮挡区域替换为打乱通道后的图像。

        # 过程3
        # sample = self.shift_transform(image=sample['image'], mask = sample['mask'])  # 使用shift_transform变换对图像和掩码进行随机平移、缩放和旋转。
        img = sample['image']
        mask = sample['mask'] 

        sample = self.transform(image=img, mask = mask)  # 使用 transform 变换对图像和掩码进行预处理。

        return sample['image'].float(), sample['mask'].float()  # 返回预处理后的图像和掩码。
