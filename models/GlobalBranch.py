from datetime import datetime
import time
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import utils
from utils.validation import validation
from utils.losses import VGGLoss
import cv2
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

def charbonnier_loss(restored, target):
    """
    计算恢复图像与目标图像之间的 Charbonnier 损失。

    Charbonnier 损失是一种平滑的 L1 损失，用于避免 L2 损失在零点处的梯度爆炸问题。
    它通过在平方差上加上一个小的常数 eps 来保证数值稳定性。

    参数:
    restored (torch.Tensor): 恢复后的图像张量。
    target (torch.Tensor): 目标图像张量。

    返回:
    torch.Tensor: 计算得到的 Charbonnier 损失值。
    """
    # 定义一个小的常数，用于避免平方根下的值为零，提高数值稳定性
    eps = 1e-3
    diff = restored - target
    loss = torch.mean(torch.sqrt((diff * diff) + (eps*eps)))
    return loss

# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft


class FocalFrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
        low_freq_weight (float): the weight for low frequency components. Default: 2.0
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False, low_freq_weight=2.0):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix
        self.low_freq_weight = low_freq_weight

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

            # Adjust low frequency weights
            _, _, _, h, w= weight_matrix.shape
            center_h, center_w = h // 2, w // 2
            low_freq_mask = torch.zeros((h, w), dtype=torch.bool, device=weight_matrix.device)
            low_freq_mask[center_h - h // 4:center_h + h // 4, center_w - w // 4:center_w + w // 4] = True
            weight_matrix[:, :, :, low_freq_mask] *= self.low_freq_weight
            weight_matrix = torch.clamp(weight_matrix, min=0.0, max=1.0)  # 限制范围

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight



class BrightnessLoss(nn.Module):
    def __init__(self):
        super(BrightnessLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        # 确保输入是 RGB 格式 (N, 3, H, W)
        assert pred.shape == target.shape, "预测和目标的形状必须相同"
        assert pred.shape[1] == 3, "输入必须是 RGB 图像 (N, 3, H, W)"

        # 归一化到 [0, 255] 范围 (防止 cv2 转换问题)
        pred = (pred.clamp(0, 1) * 255).byte()
        target = (target.clamp(0, 1) * 255).byte()

        # (N, C, H, W) → (N, H, W, C) 以适应 OpenCV
        pred_np = pred.permute(0, 2, 3, 1).cpu().numpy()
        target_np = target.permute(0, 2, 3, 1).cpu().numpy()

        # 计算 V 通道
        pred_v = np.zeros((pred.shape[0], pred.shape[2], pred.shape[3]), dtype=np.float32)
        target_v = np.zeros((target.shape[0], target.shape[2], target.shape[3]), dtype=np.float32)

        for i in range(pred.shape[0]):
            pred_v[i] = cv2.cvtColor(pred_np[i], cv2.COLOR_RGB2HSV)[:, :, 2]  # 提取 V 通道
            target_v[i] = cv2.cvtColor(target_np[i], cv2.COLOR_RGB2HSV)[:, :, 2]  # 提取 V 通道

        # 转换回 Tensor 并归一化到 [0,1]
        pred_v = torch.from_numpy(pred_v).unsqueeze(1).float() / 255.0  # (N, 1, H, W)
        target_v = torch.from_numpy(target_v).unsqueeze(1).float() / 255.0  # (N, 1, H, W)

        # 计算 V 通道的 L1 损失
        brightness_loss = self.criterion(pred_v, target_v)

        return brightness_loss

class SaturationLoss(nn.Module):
    def __init__(self):
        super(SaturationLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        # 确保输入的图像是 RGB 图像 (N, 3, H, W)
        assert pred.shape == target.shape, "预测和目标的形状必须相同"
        assert pred.shape[1] == 3, "输入必须是 RGB 图像 (N, 3, H, W)"

        # 将 tensor 转换为 numpy 数组并从 (N, C, H, W) 转换为 (N, H, W, C) 格式
        pred = pred.permute(0, 2, 3, 1).detach().cpu().numpy()  # (N, H, W, C)
        target = target.permute(0, 2, 3, 1).detach().cpu().numpy()  # (N, H, W, C)

        # 将 RGB 转换为 HSV
        pred_hsv = np.zeros_like(pred, dtype=np.float32)
        target_hsv = np.zeros_like(target, dtype=np.float32)

        for i in range(pred.shape[0]):
            pred_hsv[i] = cv2.cvtColor(pred[i], cv2.COLOR_RGB2HSV)
            target_hsv[i] = cv2.cvtColor(target[i], cv2.COLOR_RGB2HSV)

        # 提取饱和度通道（S）
        pred_saturation = pred_hsv[:, :, :, 1]  # (N, H, W)
        target_saturation = target_hsv[:, :, :, 1]  # (N, H, W)

        # 转换为 PyTorch 张量，并调整维度为 (N, 1, H, W)
        pred_saturation = torch.from_numpy(pred_saturation).unsqueeze(1).float()  # (N, 1, H, W)
        target_saturation = torch.from_numpy(target_saturation).unsqueeze(1).float()  # (N, 1, H, W)

        # 计算饱和度损失
        saturation_loss = self.criterion(pred_saturation, target_saturation)

        return saturation_loss


class GlobalBranch(object):
    def __init__(self, model, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = model
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0
        self.validation = validation
        self.l1_loss = nn.L1Loss()
        self.perception_loss = VGGLoss()
        self.fft_loss = FocalFrequencyLoss()
        # self.y_loss = yChannelLoss()
        self.saturation_loss = SaturationLoss()
        self.brightness_loss = BrightnessLoss()
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.training.n_epochs, eta_min=1e-6)

    def load_ddm_ckpt(self, load_path):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)  # True
        #self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def train(self, dataset):
        cudnn.benchmark = True
        train_loader, val_loader = dataset.get_loaders()
        epoch_loss = 0
        best_psnr = 25.0  # 记录最佳 PSNR

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        old_val_psnr, old_val_ssim, old_val_ciede = self.validation(
            self.model, val_loader, self.device, self.config, save_tag=False)
        print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('***** Epoch: ', epoch, '*****')

            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)
                input = x[:, :3, :, :]
                gt = x[:, 3:6, :, :]
                
                # 模型输出
                pre, haze_recon,ffb_out = self.model(input)
                #pre= self.model(input)

                # 计算损失
                l1 = self.l1_loss(pre, gt)
                perception = self.perception_loss(pre, gt)
                fft_loss = self.fft_loss(pre, gt)
                l1_haze = self.l1_loss(haze_recon, input)

                #fre_loss = self.fft_loss(ffb_out, gt)
                
                loss = l1 + perception + fft_loss + 0.1 * l1_haze
                epoch_loss += loss.item()

                if self.step % 100 == 0:
                    self.model.eval()
                    print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:0.4f}, L1: {:.4f}, Perception: {:.4f}'.
                        format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                epoch, self.config.training.n_epochs,
                                self.step, self.config.training.snapshot_freq,
                                (epoch_loss / self.step), l1.item(), perception.item()))

                    log_path = './log/DHID.txt'
                    os.makedirs(os.path.dirname(log_path), exist_ok=True)
                    with open(log_path, 'a') as f:
                        print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:0.4f}, L1: {:.4f}, Perception: {:.4f}'.
                            format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                    epoch, self.config.training.n_epochs,
                                    self.step, self.config.training.snapshot_freq,
                                    (epoch_loss / self.step), l1.item(), perception.item()), file=f)




                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                data_start = time.time()

            # 计算验证集 PSNR
            self.model.eval()
            val_psnr, val_ssim, val_ciede = self.validation(
                self.model, val_loader, self.device, self.config, save_tag=True)
            print('{} Epoch [{:03d}/{:03d}], Loss: {:0.4f}, Val_PSNR: {:.2f}, Val_SSIM: {:.4f}'.
                format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        epoch, self.config.training.n_epochs,
                        (epoch_loss / self.step), val_psnr, val_ssim))

            log_path = './log/DHID.txt'
            with open(log_path, 'a') as f:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:0.4f}, L1: {:.4f}, Perception: {:.4f}, Val_PSNR: {:.2f}, Val_SSIM: {:.4f}, Val_CIEDE:{:.4f}'.
                    format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                            epoch, self.config.training.n_epochs,
                            self.step, self.config.training.snapshot_freq,
                            (epoch_loss / self.step), l1.item(), perception.item(),
                            val_psnr, val_ssim, val_ciede), file=f)



            # 记录最佳 PSNR，并保存模型
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                utils.logging.save_checkpoint({
                    'epoch': epoch + 1,
                    'step': self.step,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'params': self.args,
                    'config': self.config,
                    'scheduler': self.scheduler.state_dict(),
                }, filename=os.path.join('./checkpoints', self.config.data.dataset,
                                        'best_model_{:.2f}_{:.4f}'.format(val_psnr, val_ssim)))


