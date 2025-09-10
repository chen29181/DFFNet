import torch
import torch.nn as nn
import math
from einops import rearrange
import numbers
import torch.nn.functional as F
import numpy as np







class PNM(nn.Module):
    def __init__(self, channels=32):
        super(PNM, self).__init__()
        self.channels = channels
        self.inst_norm = nn.InstanceNorm2d(channels // 2)  # 使用channels参数
        self.gn =  nn.GroupNorm(num_groups=4, num_channels=channels // 2)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        mid = self.conv(x)
        mid1, mid2 = torch.chunk(mid, 2, dim=1)
        mid1 = self.inst_norm(mid1)
        mid2 = self.gn(mid2)
        mid1 = self.relu(mid1)
        mid2 = self.relu(mid2)
        out = torch.cat([mid1, mid2], dim=1)
        out = self.conv(out)
        out = out + x
        return out

class MSDEU(nn.Module):
    def __init__(self, channels=32):
        super(MSDEU, self).__init__()
        self.channels = channels
        self.pnm = PNM(channels)
        self.conv3_1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.conv3_2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.conv1_1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv1_3 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv1_9 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv1_27 = nn.Conv2d(channels, channels, kernel_size=1)

        self.identify = nn.Conv2d(channels, channels * 4, kernel_size=1)

        self.dilated_conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1)
        self.dilated_conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3)
        self.dilated_conv9 = nn.Conv2d(channels, channels, kernel_size=3, padding=9, dilation=9)
        self.dilated_conv27 = nn.Conv2d(channels, channels, kernel_size=3, padding=36, dilation=3)

        self.conv3_3 = nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1)
        self.conv3_5 = nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1)

        self.adjust_1 = nn.Conv2d(channels, channels * 4, kernel_size=1)
        self.adjust_2 = nn.Conv2d(channels * 4, channels, kernel_size=1)

        
    def forward(self, x):
        norm_x = self.pnm(x)
        initial_enhance = norm_x + self.relu(self.conv3_1(norm_x))

        conv1_in = self.conv3_2(initial_enhance)

        multi_scale_out1 = self.dilated_conv1(self.conv1_1(conv1_in))
        multi_scale_out3 = self.dilated_conv3(self.conv1_3(conv1_in))
        multi_scale_out9 = self.dilated_conv9(self.conv1_9(conv1_in))
        multi_scale_out27 = self.dilated_conv27(self.conv1_27(conv1_in))
        multi_scale_concat = torch.cat(
            [multi_scale_out1, multi_scale_out3, multi_scale_out9, multi_scale_out27], dim=1)

        multi_scale_concat = self.identify(conv1_in) + multi_scale_concat
       

        out_1 = self.relu(self.conv3_3(multi_scale_concat))

        out_2 = out_1 + self.relu(self.conv3_4(out_1))

        out_3 = self.conv3_5(out_2)

        output = self.adjust_1(initial_enhance)
        output = self.adjust_2(output)

       

        return output


###########################

#########注意力############

class GatedUnit(nn.Module):
    def __init__(self, channels):
        super(GatedUnit, self).__init__()

        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate_conv1x1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.relu_bn = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels)
        )
        self.max_avg_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.gate_conv7x7 = nn.Conv2d(channels, channels, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        feat = self.conv1x1(x)
        gate_feat = self.gate_conv1x1(feat)
        gate_feat = self.relu_bn(gate_feat)
        gate_feat = self.max_avg_pool(gate_feat)
        gate_feat = self.gate_conv7x7(gate_feat) + feat
        gate = self.sigmoid(gate_feat)  # 生成门控权重
        return gate * feat  # 门控作用
        #return gate

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FSAS(nn.Module):
    def __init__(self, dim, bias):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 8

        self.gated_unit = GatedUnit(dim)
        self.adjust_conv = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.channel_recover=nn.Conv2d(dim*4,dim*2,kernel_size=1,bias=bias)

        self.local_conv=nn.Sequential(
            nn.Conv2d(dim,dim*2,kernel_size=3,padding=1,bias=bias),
            nn.GroupNorm(num_groups=4, num_channels=dim*2),
            nn.ReLU()
        )

       

        #创建一个一维张量
        self.alpha = nn.Parameter(torch.randn(1))
        # #这个用来控制融合的贡献
        self.beta = nn.Parameter(torch.randn(1))
        
    def forward(self, x):
        hidden = self.to_hidden(x)#x(,dim,,)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)# q(,dim*2,,)
        #print(x.shape,q.shape)
        q_enhance=self.gated_unit(x)
        q_enhance = self.adjust_conv(q_enhance)
        
        q=torch.cat([q,q_enhance],dim=1)
        q=self.channel_recover(q)



        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        ########
        local_feat=self.local_conv(x)
       
        output=output+local_feat*self.alpha
        

        ############
        output = self.project_out(output)

       
        beta=torch.sigmoid(self.beta)
        return output*beta + x*(1-beta) 

#########注意力###############

####################################
class getLowHigh(nn.Module):
    def __init__(self, in_channels, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.rate_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // 8, 2, 1, bias=False),
        )

    def shift(self, x):
        '''shift FFT feature map to center'''
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(int(h / 2), int(w / 2)), dims=(2, 3))

    def unshift(self, x):
        """converse to shift operation"""
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(-int(h / 2), -int(w / 2)), dims=(2, 3))

    def fft(self, x):
        """obtain high/low-frequency features from input"""
        x = self.conv1(x)
        mask = torch.zeros(x.shape).to(x.device)

        h, w = x.shape[-2:]
        threshold = F.adaptive_avg_pool2d(x, 1)

        threshold = self.rate_conv(threshold).sigmoid()

        for i in range(mask.shape[0]):
            h_ = torch.floor(threshold[i, 0, :, :] * h).int()
            w_ = torch.floor(threshold[i, 1, :, :] * w).int()
            mask[i, :, (h - h_) // 2:(h + h_) // 2, (w - w_) // 2:(w + w_) // 2] = 1

        fft = torch.fft.fft2(x, norm='forward', dim=(-2, -1))
        fft = self.shift(fft)

        fft_high = fft * (1 - mask)
        high = self.unshift(fft_high)
        high = torch.fft.ifft2(high, norm='forward', dim=(-2, -1))
        high = torch.abs(high)

        fft_low = fft * mask
        low = self.unshift(fft_low)
        low = torch.fft.ifft2(low, norm='forward', dim=(-2, -1))
        low = torch.abs(low)#理论上用.real取实部更好。这里容易让人误解是取振幅

        return high, low #此处不是振幅图，而是实际图像的高低频部分。
        

class FourierUnit(nn.Module):
    def __init__(self, channels, groups=1):
        super(FourierUnit, self).__init__()
        self.channels = channels
        self.lowhigh = getLowHigh(channels, channels)
        # 定义振幅处理的层
        self.amplitude_conv1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, groups=groups)
        self.amplitude_relu1 = nn.LeakyReLU()
        self.amplitude_conv_mid = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=groups)
        self.amplitude_relu2 = nn.LeakyReLU()
        self.amplitude_conv2 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, groups=groups)

        # 定义相位处理的层
        self.phase_conv1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, groups=groups)
        self.phase_relu1 = nn.ReLU() #nn.LeakyReLU()
        self.phase_conv_mid = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=groups)
        self.phase_relu2 = nn.ReLU()#nn.LeakyReLU()
        self.phase_conv2 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, groups=groups)

       
        
    def forward(self, x):
        batch, c, h, w = x.shape


        # 获取高低频
        high, low =self.lowhigh.fft(x)

        # 低频处理（振幅处理）
        low_ffted = torch.fft.rfft2(low, dim=(-2, -1), norm="ortho")
        low_amplitude = torch.abs(low_ffted)

        low_amplitude_out = self.amplitude_conv1(low_amplitude)
        low_amplitude_out = self.amplitude_relu1(low_amplitude_out)
        low_amplitude_out = self.amplitude_conv_mid(low_amplitude_out)
        low_amplitude_out = self.amplitude_relu2(low_amplitude_out)
        low_amplitude_out = self.amplitude_conv2(low_amplitude_out)
        
        low_phase = torch.angle(low_ffted)
        low_transformed = low_amplitude_out * torch.exp(1j * low_phase)
        low_output = torch.fft.irfft2(low_transformed, s=(h, w), dim=(-2, -1), norm="ortho")

        
        high_ffted = torch.fft.rfft2(high, dim=(-2, -1), norm="ortho")
        high_phase = torch.angle(high_ffted)# 高频处理（相位处理）

        high_phase_out = self.phase_conv1(high_phase)
        high_phase_out = self.phase_relu1(high_phase_out)
        high_phase_out = self.phase_conv_mid(high_phase_out)
        high_phase_out = self.phase_relu2(high_phase_out)
        high_phase_out = self.phase_conv2(high_phase_out)
       

        high_amplitude = torch.abs(high_ffted)
        high_transformed = high_amplitude * torch.exp(1j * high_phase_out)
        # theta = torch.tanh(high_phase_out) * np.pi  # 映射到 [-π, π]
        # high_transformed = high_amplitude * torch.exp(1j * theta)
        high_output = torch.fft.irfft2(high_transformed, s=(h, w), dim=(-2, -1), norm="ortho")

        # 合并高低频结果
        output = low_output + high_output



        return output 

class SpatialModel(nn.Module):
    def __init__(self, channels=32):
        super(SpatialModel, self).__init__()
        self.pnm = PNM(channels)
        self.msdeu = MSDEU(channels)

    def forward(self, x):
        x = self.pnm(x)
        x = self.msdeu(x)
        return x


#################################################################################

class BasicLayer(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # 构建 FFB 块
        self.ffb_blocks = nn.ModuleList([FourierUnit(dim) for i in range(depth)])
        # 构建 SpatialModel 块
        self.spatial_blocks = nn.ModuleList([SpatialModel(channels=dim) for i in range(depth)])

    def forward_ffb(self, x):
        for ffb in self.ffb_blocks:
            x = ffb(x)
        return x

    def forward_spatial(self, x):
        for spatial in self.spatial_blocks:
            x = spatial(x)
        return x


##########################################################################
##---------- Resizing Modules ----------
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias)
        )

    def forward(self, x):
        return self.bot(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()

        modules_body = []
        modules_body.append(Down(in_channels, out_channels))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x



class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(Up, self).__init__()

        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        )

    def forward(self, x):
        return self.bot(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()

        modules_body = []
        modules_body.append(Up(in_channels, out_channels))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out

#####################################################


################################################


class BlockUNet1(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, relu=False, drop=False, bn=True):
        super(BlockUNet1, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)

        self.dropout = nn.Dropout2d(0.5)
        self.batch = nn.InstanceNorm2d(out_channels)

        self.upsample = upsample
        self.relu = relu
        self.drop = drop
        self.bn = bn

    def forward(self, x):
        if self.relu == True:
            y = F.relu(x)
        elif self.relu == False:
            y = F.leaky_relu(x, 0.2)
        if self.upsample == True:
            y = self.deconv(y)
            if self.bn == True:
                y = self.batch(y)
            if self.drop == True:
                y = self.dropout(y)

        elif self.upsample == False:
            y = self.conv(y)
            if self.bn == True:
                if y.shape[2] == 1:
                    y = y
                else:
                    y = self.batch(y)
            if self.drop == True:
                y = self.dropout(y)

        return y


class G2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(G2, self).__init__()

        self.conv = nn.Conv2d(in_channels, 8, 4, 2, 1, bias=False)
        self.layer1 = BlockUNet1(8, 16)
        self.layer2 = BlockUNet1(16, 32)

        self.dlayer2 = BlockUNet1(32, 16, True, True, True, False)
        self.dlayer1 = BlockUNet1(32, 8, True, True)
        self.relu = nn.ReLU()
        self.dconv = nn.ConvTranspose2d(16, out_channels, 4, 2, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        y1 = self.conv(x)
        y2 = self.layer1(y1)
        y3 = self.layer2(y2)

        dy3 = self.dlayer2(y3)
        concat2 = torch.cat([dy3, y2], 1)
        dy2 = self.dlayer1(concat2)
        concat1 = torch.cat([dy2, y1], 1)
        out = self.relu(concat1)
        out = self.dconv(out)
        out = self.lrelu(out)

        return F.avg_pool2d(out, (out.shape[2], out.shape[3]))



############################
class GlobalNet_Fusion(nn.Module):
    def __init__(self, in_chans=3, out_chans=3,
                 embed_dims=[24, 48, 96, 48, 24],  # 24, 48, 96, 48, 24
                 depths=[1, 1, 2, 1, 1]):  # 1, 1, 2, 1, 1
        super(GlobalNet_Fusion, self).__init__()

        self.conv1 = nn.Conv2d(in_chans, embed_dims[0], kernel_size=3, padding=1)

        # backbone
        self.layer1 = BasicLayer(dim=embed_dims[0], depth=depths[0])

        self.down1 = DownSample(embed_dims[0], embed_dims[1])

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(dim=embed_dims[1], depth=depths[1])

        self.down2 = DownSample(embed_dims[1], embed_dims[2])

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(dim=embed_dims[2], depth=depths[2])

        ##############
        self.fsas3 = FSAS(embed_dims[2] *2, bias=False)
        ###############

        self.up1 = UpSample(embed_dims[2], embed_dims[3])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])
       
       
        self.fsas1 = FSAS(embed_dims[3] *2, bias=False)
        

        self.layer4 = BasicLayer(dim=embed_dims[3], depth=depths[3])

        self.up2 = UpSample(embed_dims[3], embed_dims[4])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])

       
        self.fsas2 = FSAS(embed_dims[4] *2, bias=False)

        self.layer5 = BasicLayer(dim=embed_dims[4], depth=depths[4])

        self.conv2 = nn.Conv2d(embed_dims[4]*2, out_chans, kernel_size=3, padding=1)

        #self.fre_decoder=nn.Conv2d(embed_dims[4],out_chans,kernel_size=3,padding=1)

        self.conv_T_1 = nn.Conv2d(embed_dims[4]*2, embed_dims[4]*2, 3, 1, 1, bias=False)
        self.conv_T_2 = nn.Conv2d(embed_dims[4]*2, 1, 3, 1, 1, bias=False)
        self.ANet = G2(3, 3)

    

    def forward_features(self, x):
       
        
        original = x

        x = self.conv1(x)  # 3*3Conv

        # 频域和空域分支
        ffb_out1 = self.layer1.forward_ffb(x)
        spatial_out1 = self.layer1.forward_spatial(x)

        skip1_ffb = ffb_out1
        skip1_spatial = spatial_out1

        ffb_out1 = self.down1(ffb_out1)
        spatial_out1 = self.down1(spatial_out1)

        ffb_out2 = self.layer2.forward_ffb(ffb_out1)
        spatial_out2 = self.layer2.forward_spatial(spatial_out1)

        skip2_ffb = ffb_out2
        skip2_spatial = spatial_out2

        ffb_out2 = self.down2(ffb_out2)
        spatial_out2 = self.down2(spatial_out2)

        ffb_out3 = self.layer3.forward_ffb(ffb_out2)
        spatial_out3 = self.layer3.forward_spatial(spatial_out2)

        ####################
        # 使用 FSAS 融合
        combined = torch.cat([ffb_out3, spatial_out3], dim=1)
        combined = self.fsas3(combined)

        ffb_out3 = combined[:, :ffb_out3.size(1), :, :]
        spatial_out3 = combined[:, ffb_out3.size(1):, :, :]
        ###########################

        
        ffb_out3 = self.up1(ffb_out3)
        spatial_out3 = self.up1(spatial_out3)

        # 融合 skip connection
        ffb_out3 = self.fusion1([ffb_out3, self.skip2(skip2_ffb)]) + ffb_out3
        spatial_out3 = self.fusion1([spatial_out3, self.skip2(skip2_spatial)]) + spatial_out3

        # 经过 BasicLayer
        ffb_out4 = self.layer4.forward_ffb(ffb_out3)
        spatial_out4 = self.layer4.forward_spatial(spatial_out3)

        

        # 使用 FSAS 融合
        combined1 = torch.cat([ffb_out4, spatial_out4], dim=1)
        combined1 = self.fsas1(combined1)
        

        ffb_out4 = combined1[:, :ffb_out4.size(1), :, :]
        spatial_out4 = combined1[:, ffb_out4.size(1):, :, :]

        ffb_out4 = self.up2(ffb_out4)
        spatial_out4 = self.up2(spatial_out4)

        # 融合 skip connection
        ffb_out4 = self.fusion2([ffb_out4, self.skip1(skip1_ffb)]) + ffb_out4
        spatial_out4 = self.fusion2([spatial_out4, self.skip1(skip1_spatial)]) + spatial_out4

        # 经过 BasicLayer
        ffb_out5 = self.layer5.forward_ffb(ffb_out4)
        spatial_out5 = self.layer5.forward_spatial(spatial_out4)

        # 使用 FSAS 融合
        combined2 = torch.cat([ffb_out5, spatial_out5], dim=1)
        combined2 = self.fsas2(combined2)
        
        ffb_out = self.fre_decoder(ffb_out5)

        x = combined2
        x = self.conv2(x)

        #return x
        
        out_J=torch.clamp(x,0,1)

        out_T = self.conv_T_1(combined2)
        out_T = self.conv_T_2(out_T)
        out_T = torch.clamp(out_T, 0, 1)

        out_A = self.ANet(original)
        out_A = torch.clamp(out_A, 0, 1)

        out_I = out_T * out_J + (1 - out_T) * out_A

        #清晰图像，雾图像，频域解码图像
        return out_J,out_I


       

    def forward(self, x):
        global_img,haze_recon = self.forward_features(x)
        #global_img= self.forward_features(x)
    
        return global_img , haze_recon


if __name__ == '__main__':
    model = GlobalNet_Fusion()
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    from thop import profile
    flops, params = profile(model, inputs=(x,))
    print('Params and FLOPs are {}M/{}G'.format(params/1e6, flops/1e9))
