from .backbone.EfficientNet import EfficientNet

import torch
import numpy as np
import torch.nn as nn
from torch.fft import fft2, fftshift, ifft2, ifftshift
# from util.utils import *
import torch.nn.functional as F
# from config import getConfig
from .attention.conv_modules import BasicConv2d, DWConv, DWSConv


# cfg = getConfig()


class Frequency_Edge_Module(nn.Module):
    def __init__(self, radius, channel):
        super(Frequency_Edge_Module, self).__init__()
        self.radius = radius
        self.UAM = UnionAttentionModule(channel, only_channel_tracing=True)

        # DWS + DWConv
        self.DWSConv = DWSConv(channel, channel, kernel=3, padding=1, kernels_per_layer=1)
        self.DWConv1 = nn.Sequential(
            DWConv(channel, channel, kernel=1, padding=0, dilation=1),
            BasicConv2d(channel, channel // 4, 1),
        )
        self.DWConv2 = nn.Sequential(
            DWConv(channel, channel, kernel=3, padding=1, dilation=1),
            BasicConv2d(channel, channel // 4, 1),
        )
        self.DWConv3 = nn.Sequential(
            DWConv(channel, channel, kernel=3, padding=3, dilation=3),
            BasicConv2d(channel, channel // 4, 1),
        )
        self.DWConv4 = nn.Sequential(
            DWConv(channel, channel, kernel=3, padding=5, dilation=5),
            BasicConv2d(channel, channel // 4, 1),
        )
        self.conv = BasicConv2d(channel, 1, 1)

    def distance(self, i, j, imageSize, r):
        dis = np.sqrt((i - imageSize / 2) ** 2 + (j - imageSize / 2) ** 2)
        if dis < r:
            return 1.0
        else:
            return 0

    def mask_radial(self, img, r):
        batch, channels, rows, cols = img.shape
        mask = torch.zeros((rows, cols), dtype=torch.float32)
        for i in range(rows):
            for j in range(cols):
                mask[i, j] = self.distance(i, j, imageSize=rows, r=r)
        return mask

    def forward(self, x):
        """
        Input:
            The first encoder block representation: (B, C, H, W)
        Returns:
            Edge refined representation: X + edge (B, C, H, W)
        """
        x_fft = fft2(x, dim=(-2, -1))
        x_fft = fftshift(x_fft)

        # Mask -> low, high separate
        mask = self.mask_radial(img=x, r=self.radius).cuda()
        high_frequency = x_fft * (1 - mask)
        x_fft = ifftshift(high_frequency)
        x_fft = ifft2(x_fft, dim=(-2, -1))
        x_H = torch.abs(x_fft)

        x_H, _ = self.UAM.Channel_Tracer(x_H)
        edge_maks = self.DWSConv(x_H)
        skip = edge_maks.clone()

        edge_maks = torch.cat([self.DWConv1(edge_maks), self.DWConv2(edge_maks),
                               self.DWConv3(edge_maks), self.DWConv4(edge_maks)], dim=1) + skip
        edge = torch.relu(self.conv(edge_maks))

        x = x + edge  # Feature + Masked Edge information

        return x, edge


class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        # 先经过3个卷积层生成3个新特征图B C D （尺寸不变）
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # α尺度系数初始化为0，并逐渐地学习分配到更大的权重

        self.softmax = nn.Softmax(dim=-1)  # 对每一行进行softmax

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × (H×W) × (H×W)
        """
        m_batchsize, C, height, width = x.size()
        # B -> (N,C,HW) -> (N,HW,C)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # C -> (N,C,HW)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        # BC，空间注意图 -> (N,HW,HW)
        energy = torch.bmm(proj_query, proj_key)
        # S = softmax(BC) -> (N,HW,HW)
        attention = self.softmax(energy)
        # D -> (N,C,HW)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        # DS -> (N,C,HW)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # torch.bmm表示批次矩阵乘法
        # output -> (N,C,H,W)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))  # β尺度系数初始化为0，并逐渐地学习分配到更大的权重
        self.softmax = nn.Softmax(dim=-1)  # 对每一行进行softmax

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × C × C
        """
        m_batchsize, C, height, width = x.size()
        # A -> (N,C,HW)
        proj_query = x.view(m_batchsize, C, -1)
        # A -> (N,HW,C)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        # 矩阵乘积，通道注意图：X -> (N,C,C)
        energy = torch.bmm(proj_query, proj_key)
        # 这里实现了softmax用最后一维的最大值减去了原始数据，获得了一个不是太大的值
        # 沿着最后一维的C选择最大值，keepdim保证输出和输入形状一致，除了指定的dim维度大小为1
        # expand_as表示以复制的形式扩展到energy的尺寸
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy

        attention = self.softmax(energy_new)
        # A -> (N,C,HW)
        proj_value = x.view(m_batchsize, C, -1)
        # XA -> （N,C,HW）
        out = torch.bmm(attention, proj_value)
        # output -> (N,C,H,W)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class GlobalAvgPool(nn.Module):
    def __init__(self, flatten=False):
        super(GlobalAvgPool, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


class Attention(nn.Module):
    def __init__(self, n_channels):
        super(Attention, self).__init__()
        self.GAP = GlobalAvgPool()
        self.confidence_ratio = 0.1
        self.bn = nn.BatchNorm2d(n_channels)
        self.norm = nn.Sequential(
            nn.BatchNorm2d(n_channels),
            nn.Dropout3d(self.confidence_ratio)
        )

        inter_channels = n_channels // 2

        self.conv5a = nn.Sequential(nn.Conv2d(n_channels, n_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(n_channels), nn.ReLU())
        self.conv5c = nn.Sequential(nn.Conv2d(n_channels, n_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(n_channels), nn.ReLU())

        self.conv51 = nn.Sequential(nn.Conv2d(n_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels), nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(n_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels), nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(56, 1, 1))

        self.spatial_q = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.spatial_k = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.spatial_v = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.sa = PAM_Module(n_channels)  # 空间注意力模块
        self.sc = CAM_Module(n_channels)  # 通道注意力模块
        self.sigmoid = nn.Sigmoid()

    def CamAndPam(self, x):
        # 经过一个1×1卷积降维后，再送入通道注意力模块
        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)

        sa_feat = self.sa(sc_feat)
        # 先经过一个卷积后，再使用有dropout的1×1卷积输出指定的通道数
        sa_conv = self.conv51(sa_feat)

        return sa_conv

    def PamAndCam(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        # 先经过一个卷积后，再使用有dropout的1×1卷积输出指定的通道数

        sc_feat = self.sc(sa_feat)
        # 先经过一个卷积后，再使用有dropout的1×1卷积输出指定的通道数
        sc_conv = self.conv52(sc_feat)

        return sc_conv

    def forward(self, x):
        sa_conv = self.CamAndPam(x)
        sc_conv = self.PamAndCam(x)
        x_drop = torch.cat((sa_conv, sc_conv), 1)

        q = self.spatial_q(x_drop).squeeze(1)
        k = self.spatial_k(x_drop).squeeze(1)
        v = self.spatial_v(x_drop).squeeze(1)

        # softmax(Q*K^T)
        QK_T = torch.matmul(q, k.transpose(1, 2))
        alpha = F.softmax(QK_T, dim=-1)

        output = torch.matmul(alpha, v).unsqueeze(1) + v.unsqueeze(1)

        return output


class UnionAttentionModule(nn.Module):
    def __init__(self, n_channels, only_channel_tracing=False):
        super(UnionAttentionModule, self).__init__()
        self.GAP = GlobalAvgPool()
        self.confidence_ratio = 0.1
        self.bn = nn.BatchNorm2d(n_channels)
        self.norm = nn.Sequential(
            nn.BatchNorm2d(n_channels),
            nn.Dropout3d(self.confidence_ratio)
        )
        self.channel_q = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.channel_k = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.channel_v = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, stride=1,
                                   padding=0, bias=False)

        self.fc = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, stride=1,
                            padding=0, bias=False)

        if only_channel_tracing == False:
            self.spatial_q = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1, stride=1,
                                       padding=0, bias=False)
            self.spatial_k = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1, stride=1,
                                       padding=0, bias=False)
            self.spatial_v = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1, stride=1,
                                       padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def masking(self, x, mask):
        mask = mask.squeeze(3).squeeze(2)
        threshold = torch.quantile(mask, self.confidence_ratio, dim=-1, keepdim=True)
        mask[mask <= threshold] = 0.0
        mask = mask.unsqueeze(2).unsqueeze(3)
        mask = mask.expand(-1, x.shape[1], x.shape[2], x.shape[3]).contiguous()
        masked_x = x * mask

        return masked_x

    def Channel_Tracer(self, x):
        avg_pool = self.GAP(x)
        x_norm = self.norm(avg_pool)

        q = self.channel_q(x_norm).squeeze(-1)
        k = self.channel_k(x_norm).squeeze(-1)
        v = self.channel_v(x_norm).squeeze(-1)

        # softmax(Q*K^T)
        QK_T = torch.matmul(q, k.transpose(1, 2))
        alpha = F.softmax(QK_T, dim=-1)

        # a*v
        att = torch.matmul(alpha, v).unsqueeze(-1)
        att = self.fc(att)
        att = self.sigmoid(att)

        output = (x * att) + x
        alpha_mask = att.clone()

        return output, alpha_mask

    def forward(self, x):
        X_c, alpha_mask = self.Channel_Tracer(x)
        X_c = self.bn(X_c)
        x_drop = self.masking(X_c, alpha_mask)

        q = self.spatial_q(x_drop).squeeze(1)
        k = self.spatial_k(x_drop).squeeze(1)
        v = self.spatial_v(x_drop).squeeze(1)

        # softmax(Q*K^T)
        QK_T = torch.matmul(q, k.transpose(1, 2))
        alpha = F.softmax(QK_T, dim=-1)

        output = torch.matmul(alpha, v).unsqueeze(1) + v.unsqueeze(1)

        return output


class aggregation(nn.Module):
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel[2], channel[1], 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel[2], channel[0], 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel[1], channel[0], 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel[2], channel[2], 3, padding=1)
        self.conv_upsample5 = BasicConv2d(channel[2] + channel[1], channel[2] + channel[1], 3, padding=1)

        self.conv_concat2 = BasicConv2d((channel[2] + channel[1]), (channel[2] + channel[1]), 3, padding=1)
        self.conv_concat3 = BasicConv2d((channel[0] + channel[1] + channel[2]),
                                        (channel[0] + channel[1] + channel[2]), 3, padding=1)
        # print(channel[0] + channel[1] + channel[2])
        self.selfAtt = Attention(channel[0] + channel[1] + channel[2])



    def forward(self, e4, e3, e2):
        e4_1 = e4
        e3_1 = self.conv_upsample1(self.upsample(e4)) * e3
        e2_1 = self.conv_upsample2(self.upsample(self.upsample(e4))) \
               * self.conv_upsample3(self.upsample(e3)) * e2

        e3_2 = torch.cat((e3_1, self.conv_upsample4(self.upsample(e4_1))), 1)
        e3_2 = self.conv_concat2(e3_2)

        e2_2 = torch.cat((e2_1, self.conv_upsample5(self.upsample(e3_2))), 1)
        x = self.conv_concat3(e2_2)

        output = self.selfAtt(x)

        return output


class FCM(nn.Module):
    def __init__(self, channel, kernel_size):
        super(FCM, self).__init__()
        self.channel = channel
        out_channel = channel // 8
        self.DWSConv = DWSConv(channel, channel // 2, kernel=kernel_size, padding=1, kernels_per_layer=1)
        self.DWConv1 = nn.Sequential(
            DWConv(channel // 2, channel // 2, kernel=1, padding=0, dilation=1),
            BasicConv2d(channel // 2, channel // 8, 1),
            # ChannelAttention(out_channel),
        )
        self.DWConv2 = nn.Sequential(
            DWConv(channel // 2, channel // 2, kernel=3, padding=1, dilation=1),
            BasicConv2d(channel // 2, channel // 8, 1),
            # ChannelAttention(out_channel),
        )
        self.DWConv3 = nn.Sequential(
            DWConv(channel // 2, channel // 2, kernel=3, padding=3, dilation=3),
            BasicConv2d(channel // 2, channel // 8, 1),
            # ChannelAttention(out_channel),
        )
        self.DWConv4 = nn.Sequential(
            DWConv(channel // 2, channel // 2, kernel=3, padding=5, dilation=5),
            BasicConv2d(channel // 2, channel // 8, 1),
            # ChannelAttention(out_channel),
        )
        self.conv1 = BasicConv2d(channel // 2, 1, 1)

    def forward(self, decoder_map, encoder_map):
        """
        Args:
            decoder_map: decoder representation (B, 1, H, W).
            encoder_map: encoder block output (B, C, H, W).
        Returns:
            decoder representation: (B, 1, H, W)
        """
        mask_bg = -1 * torch.sigmoid(decoder_map) + 1  # Sigmoid & Reverse

        mask_ob = torch.sigmoid(decoder_map)  # object attention
        x = mask_ob.expand(-1, self.channel, -1, -1).mul(encoder_map)

        edge = mask_bg.clone()
        edge[edge > 0.93] = 0
        x = x + (edge * encoder_map)

        x = self.DWSConv(x)
        skip = x.clone()
        x = torch.cat([self.DWConv1(x), self.DWConv2(x), self.DWConv3(x), self.DWConv4(x)], dim=1) + skip
        x = torch.relu(self.conv1(x))

        return x + decoder_map

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()

        self.fc1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        avg = self.fc2(self.relu(self.fc1(avg)))
        return self.sigmoid(avg) * x


class RFBC_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFBC_Block, self).__init__()
        # 添加BN层
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3),
            ChannelAttention(out_channel)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5),
            ChannelAttention(out_channel)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7),
            ChannelAttention(out_channel)
        )

        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_cat = self.conv_cat(x_cat)

        x_cat = self.bn(x_cat)

        x = self.relu(x_cat + self.conv_res(x))
        return x


class TRACER(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = EfficientNet.from_pretrained(f'efficientnet-b5', advprop=True)
        self.block_idx, self.channels = get_model_shape()
        # Receptive Field Blocks
        channels = [32, 64, 128]


        self.rfb2 = RFBC_Block(self.channels[1], channels[0])
        self.rfb3 = RFBC_Block(self.channels[2], channels[1])
        self.rfb4 = RFBC_Block(self.channels[3], channels[2])


        # Multi-level aggregation
        self.agg = aggregation(channels)


        # FCM

        self.FCM2 = FCM(channel=self.channels[1], kernel_size=3)
        self.FCM1 = FCM(channel=self.channels[0], kernel_size=3)

    def forward(self, inputs):
        B, C, H, W = inputs.size()

        # EfficientNet backbone Encoder
        x = self.model.initial_conv(inputs)
        features, edge = self.model.get_blocks(x, H, W)

        x3_rfb = self.rfb2(features[1])
        x4_rfb = self.rfb3(features[2])
        x5_rfb = self.rfb4(features[3])

        D_0 = self.agg(x5_rfb, x4_rfb, x3_rfb)

        ds_map0 = F.interpolate(D_0, scale_factor=8, mode='bilinear')

        D_1 = self.FCM2(D_0, features[1])
        ds_map1 = F.interpolate(D_1, scale_factor=8, mode='bilinear')

        ds_map = F.interpolate(D_1, scale_factor=2, mode='bilinear')
        D_2 = self.FCM1(ds_map, features[0])
        ds_map2 = F.interpolate(D_2, scale_factor=4, mode='bilinear')

        final_map = (ds_map2 + ds_map1 + ds_map0) / 3
        return torch.sigmoid(final_map), torch.sigmoid(edge), \
               (torch.sigmoid(ds_map0), torch.sigmoid(ds_map1), torch.sigmoid(ds_map2))



def get_model_shape():
    block_idx = [7, 12, 26, 38]
    channels = [40, 64, 176, 512]
    return block_idx, channels


def build_model():
    newmodel = TRACER()
    total = sum([param.nelement() for param in newmodel.parameters()])
    print('  + Number of params: %.4fM' % (total / 1e6))
    return newmodel
