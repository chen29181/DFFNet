# import argparse
# import os
# import random
# import socket
# import yaml
# import torch
# import torch.backends.cudnn as cudnn
# import numpy as np
# import torchvision
# from torch import nn

# import models
# import datasets
# import torchvision.utils as utils
# from models import GlobalNet_Fusion, GlobalBranch
# import glob
# import os
# import re
# import time

# import numpy as np
# import torch
# import torch.nn.functional as F
# import torchvision.utils as utils
# from math import log10
# from skimage import measure
# import cv2
# from skimage import img_as_float32
# import skimage
# import cv2
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr
# from skimage.metrics import structural_similarity as compare_ssim
# from skimage.color import deltaE_ciede2000 as compare_ciede

# import pdb
# from math import exp
# from torch.autograd import Variable


# def to_psnr(dehaze, gt):
#     mse = F.mse_loss(dehaze, gt, reduction='none')
#     mse_split = torch.split(mse, 1, dim=0)
#     mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

#     intensity_max = 1.0
#     psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
#     return psnr_list

# def ssim(img1, img2, window_size=11, size_average=True):
#     img1=torch.clamp(img1,min=0,max=1)
#     img2=torch.clamp(img2,min=0,max=1)
#     (_, channel, _, _) = img1.size()
#     window = create_window(window_size, channel)
#     if img1.is_cuda:
#         window = window.cuda(img1.get_device())
#     window = window.type_as(img1)
#     return _ssim(img1, img2, window, window_size, channel, size_average)

# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
#     return gauss / gauss.sum()

# def create_window(window_size, channel):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
#     return window

# def _ssim(img1, img2, window, window_size, channel, size_average=True):
#     mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

#     if size_average:
#         return [ssim_map.mean()]
#     else:
#         return [ssim_map.mean(1).mean(1).mean(1)]

# def calc_psnr(im1, im2):
#     # tensor转numpy
#     im1 = im1[0].view(im1.shape[2], im1.shape[3], 3).detach().cpu().numpy()
#     im2 = im2[0].view(im2.shape[2], im2.shape[3], 3).detach().cpu().numpy()

#     # 直接在整个图像上计算 PSNR
#     ans = [compare_psnr(im1, im2)]
#     return ans

# def calc_ssim(im1, im2):
#     im1 = im1[0].view(im1.shape[2], im1.shape[3], 3).detach().cpu().numpy()
#     im2 = im2[0].view(im2.shape[2], im2.shape[3], 3).detach().cpu().numpy()

#     # 直接在整个图像上计算 SSIM
#     ans = [compare_ssim(im1, im2, multichannel=True)]
#     return ans

    
# def calc_ciede2000(im1, im2):
#     #tensor转numpy
#     im1 = im1[0].view(im1.shape[2], im1.shape[3], 3).detach().cpu().numpy()
#     im2 = im2[0].view(im2.shape[2], im2.shape[3], 3).detach().cpu().numpy()

#     #numpy数据类型转换为float32
#     im1 =img_as_float32(im1)
#     im2 = img_as_float32(im2)
#     # im1 = im1.astype(np.float32)
#     # im2 = im2.astype(np.float32)

#     in_lab = cv2.cvtColor(im1, cv2.COLOR_RGB2Lab)
#     gt_lab = cv2.cvtColor(im2, cv2.COLOR_RGB2Lab)
#     ans = [np.average(compare_ciede(gt_lab, in_lab))]
#     return ans


# def to_ssim_skimage(pred_image, gt):
#     pred_image_list = torch.split(pred_image, 1, dim=0)
#     gt_list = torch.split(gt, 1, dim=0)

#     pred_image_list_np = [pred_image_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
#     gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
#     ssim_list = [measure.compare_ssim(pred_image_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(pred_image_list))]

#     return ssim_list


# def parse_args_and_config():
#     parser = argparse.ArgumentParser(description='Testing Global Branch Models')
#     parser.add_argument("--config", default='DHID.yml', type=str,
#                         help="Path to the config file")
#     parser.add_argument('--resume', default='/home/cz/DFFNet/checkpoints/3个fsas,没有调整融合后的权重/moderate/best_model_26.04_0.8904.pth.tar', type=str,
#                         help='Path for the diffusion model checkpoint to load for evaluation')
#     parser.add_argument("--test_set", type=str, default='results/images/',
#                         help="restoration test set options: ['DHID','ERICE']")
#     parser.add_argument("--image_folder", default='val/', type=str,
#                         help="Location to save restored images")
#     parser.add_argument('--seed', default=42, type=int, metavar='N',
#                         help='Seed for initializing training (default: 61)')
#     args = parser.parse_args()

#     with open(os.path.join("configs", args.config), "r") as f:
#         config = yaml.safe_load(f)
#     new_config = dict2namespace(config)

#     return args, new_config


# def dict2namespace(config):
#     namespace = argparse.Namespace()
#     for key, value in config.items():
#         if isinstance(value, dict):
#             new_value = dict2namespace(value)
#         else:
#             new_value = value
#         setattr(namespace, key, new_value)
#     return namespace


# def main():
#     args, config = parse_args_and_config()

#     # setup device to run
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     print("Using device: {}".format(device))
#     config.device = device

#     if torch.cuda.is_available():
#         print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

#     # set random seed
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(args.seed)
#     torch.backends.cudnn.benchmark = True

#     # data loading
#     print("=> using dataset '{}'".format(config.data.dataset))
#     DATASET = datasets.__dict__[config.data.dataset](config)
#     _, val_loader = DATASET.get_loaders()

#     # create model
#     print("=> creating global model with wrapper...")
#     model = GlobalNet_Fusion()
#     Branch = GlobalBranch(model, args, config)
#     if os.path.isfile(args.resume):
#         Branch.load_ddm_ckpt(args.resume)
#         model.eval()
#     else:
#         print('Pre-trained global model path is missing!')

#     psnr_list, ssim_list, ciede_list = [], [], []
#     for i, (x, y) in enumerate(val_loader):
#         with torch.no_grad():
#             start_time = time.time()
#             input = x[:, :3, :, :].to(device)
#             gt = x[:, 3:6, :, :].to(device)
#             # diff_img = x[:, 6:, :, :].to(device)

#             #pred_image = model(input, diff_img)
#             pred_image,_ ,_= model(input)
#             torch.cuda.synchronize()
#             inference_time = time.time() - start_time
#             print(inference_time)

#         # --- Calculate the average PSNR --- #
#         psnr_list.extend(to_psnr(pred_image, gt))

#         # --- Calculate the average SSIM --- #
#         ssim_list.extend(ssim(pred_image, gt))

#         # --- Calculate the average ciede2000 --- #
#         ciede_list.extend(calc_ciede2000(pred_image, gt))

#         pred_image_images = torch.split(pred_image, 1, dim=0)
#         batch_num = len(pred_image_images)

#         for ind in range(batch_num):
#             utils.save_image(pred_image_images[ind],
#                                  '{}{}/{}.jpg'.format(args.image_folder, config.data.dataset, y[ind]))

#     avr_psnr = sum(psnr_list) / len(psnr_list)
#     avr_ssim = sum(ssim_list) / len(ssim_list)
#     avr_ciede = sum(ciede_list) / len(ciede_list)
#     print('Current Metrics: \nPSNR: {:.3f}, \nSSIM: {:.5f}, \nCIEDE2000: {:.5f}'.format(avr_psnr, avr_ssim, avr_ciede))

# if __name__ == '__main__':
#     main()

import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torch import nn

import models
import datasets
import torchvision.utils as utils
from models import GlobalNet_Fusion, GlobalBranch
import glob
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from math import log10
from skimage import measure
import cv2
from skimage import img_as_float32
import skimage
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.color import deltaE_ciede2000 as compare_ciede

from torch.autograd import Variable


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Testing Global Branch Models')
    parser.add_argument("--config", default='DHID.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--resume', default='/home/cz/DFFNet/checkpoints/moderate_13927/best_model_25.83_0.8940.pth.tar', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--test_set", type=str, default='results/images/',
                        help="restoration test set options: ['DHID','ERICE']")
    parser.add_argument("--image_folder", default='val/', type=str,
                        help="Location to save restored images")
    parser.add_argument('--seed', default=42, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    # 设备设置
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    # 随机种子设置
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # 数据加载（不计时）
    print("=> using dataset '{}'".format(config.data.dataset))
    DATASET = datasets.__dict__[config.data.dataset](config)
    _, val_loader = DATASET.get_loaders()

    # 模型加载（不计时）
    print("=> creating global model with wrapper...")
    model = GlobalNet_Fusion()
    Branch = GlobalBranch(model, args, config)
    if os.path.isfile(args.resume):
        Branch.load_ddm_ckpt(args.resume)
        model.eval()
        model.to(device)
    else:
        print('Pre-trained global model path is missing!')
        return

    # 模型预热（排除GPU初始化耗时）
    print("=> 预热模型中...")
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= 5:  # 预热前5个batch
                break
            input = x[:, :3, :, :].to(device)
            _ ,_ ,_= model(input)  # 执行一次推理但不计时
    torch.cuda.empty_cache()

    # 推理计时统计变量
    total_inference_time = 0.0
    total_images = 0

    print("=> 开始推理计时...")
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            # 数据准备（不计时）
            input = x[:, :3, :, :].to(device)
            gt = x[:, 3:6, :, :].to(device)  # 仅用于后续保存，不参与计时
            batch_size = input.size(0)

            # 确保数据加载完成（同步GPU）
            torch.cuda.synchronize()
            start_time = time.time()

            # 仅对模型推理过程计时
            pred_image, _, _ = model(input)

            # 确保推理完成（同步GPU）
            torch.cuda.synchronize()
            inference_time = time.time() - start_time

            # 累加统计
            total_inference_time += inference_time
            total_images += batch_size

            #保存结果（不计时）
            pred_image_images = torch.split(pred_image, 1, dim=0)
            for ind in range(batch_size):
                utils.save_image(
                    pred_image_images[ind],
                    '{}{}/{}.jpg'.format(args.image_folder, config.data.dataset, y[ind])
                )

            # 打印进度
            if (i + 1) % 10 == 0:
                current_avg = total_inference_time / total_images
                print(f'处理进度: [{i+1}/{len(val_loader)}]\t'
                      f'当前平均单张耗时: {current_avg:.4f}秒')

    # 计算并输出最终统计结果
    avg_inference_time = total_inference_time / total_images
    print("\n===== 推理时间统计 =====")
    print(f'总图像数量: {total_images}张')
    print(f'总推理时间: {total_inference_time:.2f}秒')
    print(f'平均单张推理时间: {avg_inference_time:.4f}秒')


if __name__ == '__main__':
    main()