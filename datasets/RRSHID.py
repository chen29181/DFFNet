import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import random
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF

class RRSHID:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self):
        print("=> evaluating RRSHID set...")
       

        train_dataset = RRSHIDDataset(dir=os.path.join(self.config.data.data_dir, 'train'), val=False, transforms=self.transforms)
        val_dataset = RRSHIDDataset(dir=os.path.join(self.config.data.data_dir, 'val'), val=True, transforms=self.transforms)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader




class RRSHIDDataset(torch.utils.data.Dataset):
    def __init__(self, dir, val, transforms):
        super().__init__()

        self.val = val
        RRSHID_dir = dir
        input_names, gt_names = [], []

        # 读取 hazy（模糊）和 clear（清晰）图像
        RRSHID_inputs = os.path.join(RRSHID_dir, 'hazy')
        RRSHID_gt = os.path.join(RRSHID_dir, 'clear')

        input_images = [f for f in listdir(RRSHID_inputs) if isfile(os.path.join(RRSHID_inputs, f))]
        gt_images = [f for f in listdir(RRSHID_gt) if isfile(os.path.join(RRSHID_gt, f))]

        # 直接根据文件名匹配 hazy 和 clear 图像
        for hazy in input_images:
            if hazy in gt_images:  # 确保 hazy 图像和 clear 图像一一对应
                input_names.append(os.path.join(RRSHID_inputs, hazy))
                gt_names.append(os.path.join(RRSHID_gt, hazy))  # 直接使用相同的名称

        print(f"Matched {len(input_names)} hazy images with clear images.")

        # 随机打乱数据顺序
        data_pairs = list(zip(input_names, gt_names))
        random.shuffle(data_pairs)
        self.input_names, self.gt_names = zip(*data_pairs)

        self.transforms = transforms
        self.crop_size = 256 if not self.val else None  # 仅在训练时裁剪

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        img_id = os.path.splitext(os.path.basename(input_name))[0]  # 获取文件名（去掉路径和扩展名）

        input_img = PIL.Image.open(input_name)
        gt_img = PIL.Image.open(gt_name).convert('RGB')

        # 统一调整大小（保证输入大小一致）
        wd_new, ht_new = input_img.size
        if ht_new > wd_new and ht_new > 1024:
            wd_new = int(np.ceil(wd_new * 1024 / ht_new))
            ht_new = 1024
        elif ht_new <= wd_new and wd_new > 1024:
            ht_new = int(np.ceil(ht_new * 1024 / wd_new))
            wd_new = 1024
        wd_new = int(16 * np.ceil(wd_new / 16.0))
        ht_new = int(16 * np.ceil(ht_new / 16.0))
        input_img = input_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
        gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)

        # # 进行裁切
        # if self.crop_size is not None:
        #     i, j, h, w = tfs.RandomCrop.get_params(input_img, output_size=(self.crop_size, self.crop_size))
        #     input_img = FF.crop(input_img, i, j, h, w)
        #     gt_img = FF.crop(gt_img, i, j, h, w)

        return torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0), img_id

    def __getitem__(self, index):
        return self.get_images(index)

    def __len__(self):
        return len(self.input_names)
