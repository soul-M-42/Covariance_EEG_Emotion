import torch
import pytorch_lightning as pl
import torch.nn as nn
import os
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
import random
from utils_new import stratified_layerNorm
from MLLA_test import MLLA_BasicLayer

class sFilter(nn.Module):
    def __init__(self, dim_in, n_channs, sFilter_timeLen, multiFact):
        super().__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_in*multiFact, (n_channs, sFilter_timeLen), groups=dim_in)
        
    def forward(self, x):
        # x.shape = [B, n_c, dim, T]
        x = x.permute(0, 2, 1, 3)
        # x.shape = [B, dim, n_c, T]
        # out = self.conv1(x)
        out = F.relu(self.conv1(x))
        # x.shape = [B, dim, 1, T]
        # out = stratified_layerNorm(out, int(out.shape[0]/2))
        out = out.permute(0, 2, 1, 3)
        return out

class DimAttentionModule(nn.Module):
    def __init__(self, n_c, dim, seg_att):
        super(DimAttentionModule, self).__init__()
        n_msFilters_total = n_c * dim
        self.seg_att = seg_att
        # 使用padding让卷积操作的输出时间维度保持不变
        self.att_conv = nn.Conv2d(n_msFilters_total, n_msFilters_total, (1, seg_att), padding=(0, seg_att // 2), groups=n_msFilters_total)
        self.att_pool = nn.AvgPool2d((1, seg_att), stride=1, padding=(0, seg_att // 2))
        self.att_pointConv = nn.Conv2d(n_msFilters_total, n_msFilters_total, (1, 1))

    def forward(self, x):
        # [B, n_c, dim, T] -> [B, n_c * dim, T]
        b, n_c, dim, T = x.shape
        x = x.reshape(b, n_c * dim, 1, T)
        x_ori = x
        # 卷积操作
        att_w = F.relu(self.att_conv(x))  # [B, n_c * dim, 1, T]
        
        # 平均池化（生成每个通道的全局特征）
        att_w = self.att_pool(att_w)  # [B, n_c * dim, 1, T]
        att_w = self.att_pointConv(att_w)
        att_w = F.softmax(att_w, dim=1)
        
        # 加权输出
        x = att_w * F.relu(x_ori)
        x = x.reshape(b, n_c, dim, T)
        
        return x

class Clisa_Proj(nn.Module):

    def __init__(self, n_dim_in, avgPoolLen=3, multiFact=2, timeSmootherLen=6):
        super().__init__()
        self.avgpool = nn.AvgPool2d((1, avgPoolLen))
        self.timeConv1 = nn.Conv2d(n_dim_in, n_dim_in * multiFact, (1, timeSmootherLen), groups=n_dim_in)
        self.timeConv2 = nn.Conv2d(n_dim_in * multiFact, n_dim_in * multiFact * multiFact, (1, timeSmootherLen), groups=n_dim_in * multiFact)
        
    def forward(self, x):
        out = x
        [B, C, out_dim, T] = out.shape
        out = out.reshape(B, C*out_dim, 1, T)
        # for c_i in range(C):
        #     data_channel_i = out[:, c_i, :, :].unsqueeze(1)
        #     data_channel_i = self.avgpool(data_channel_i)    # B*1*n_dim*t_pool
        #     data_channel_i = stratified_layerNorm(data_channel_i, int(data_channel_i.shape[0]/2))
        #     data_channel_i = F.relu(self.timeConv1(data_channel_i))
        #     data_channel_i = F.relu(self.timeConv2(data_channel_i))          #B*(n_dim*multiFact*multiFact)*1*t_pool
        #     data_channel_i = stratified_layerNorm(data_channel_i, int(data_channel_i.shape[0]/2))     
        #     print(data_channel_i.shape)

        out = self.avgpool(out)    # B*1*n_dim*t_pool
        out = stratified_layerNorm(out, int(out.shape[0]/2))
        out = F.relu(self.timeConv1(out))
        out = F.relu(self.timeConv2(out))          #B*(n_dim*multiFact*multiFact)*1*t_pool
        out = stratified_layerNorm(out, int(out.shape[0]/2))
        out = out.reshape(out.shape[0], -1)
        out = F.normalize(out, dim=1)
        return out
    
class Channel_mlp(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.fc = nn.Linear(c_in, c_out)
        
    def forward(self, x):
        # Shape: [Batch, n_channels, N, out_dim]
        [Batch, n_channels, N, out_dim] = x.shape
        x = x.permute(0, 2, 3, 1)
        # Shape: [Batch, N, out_dim, n_channels]
        x = F.relu(self.fc(x))
        x = x.permute(0, 2, 3, 1)
        # Shape: [Batch, out_dim, n_channels, N]
        return x
    
def to_patch(data, patch_size=50, stride=25):
    batchsize, timelen, dim = data.shape
    num_patches = (timelen - patch_size) // stride + 1
    patches = torch.zeros((batchsize, num_patches, patch_size)).to('cuda')
    for i in range(num_patches):
        start_idx = i * stride
        end_idx = start_idx + patch_size
        patches[:, i, :] = data[:, start_idx:end_idx, 0]
    return patches

class Replace_Encoder(nn.Module):
    # 配置说明 125Hz采样率基线  使用参数  dilation_array=[1,3,6,12]      seg_att = 15  avgPoolLen = 15  timeSmootherLen=3 mslen = 2,3   如果频率变化请在基线上乘以相应倍数
    def __init__(self, n_timeFilters, timeFilterLen, n_msFilters, msFilter_timeLen, n_channs=64, dilation_array=np.array([1,6,12,24]), seg_att=30, avgPoolLen = 30,
                  timeSmootherLen=6, multiFact=2, stratified=[], activ='softmax', temp=1.0, saveFea=True, has_att=True, extract_mode='me', global_att=False):
        super().__init__()
        self.stratified = stratified
        self.msFilter_timeLen = msFilter_timeLen
        self.activ = activ
        self.temp = temp
        self.dilation_array = np.array(dilation_array)   
        self.saveFea = saveFea
        self.has_att = has_att
        self.extract_mode = extract_mode
        self.global_att = global_att
        

        # time and spacial conv
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen), padding=(0, (timeFilterLen-1)//2))
        self.msConv1 = nn.Conv2d(n_timeFilters, n_timeFilters*n_msFilters, (n_channs, msFilter_timeLen), groups=n_timeFilters)
        self.msConv2 = nn.Conv2d(n_timeFilters, n_timeFilters*n_msFilters, (n_channs, msFilter_timeLen), dilation=(1,self.dilation_array[1]), groups=n_timeFilters)
        self.msConv3 = nn.Conv2d(n_timeFilters, n_timeFilters*n_msFilters, (n_channs, msFilter_timeLen), dilation=(1,self.dilation_array[2]), groups=n_timeFilters)
        self.msConv4 = nn.Conv2d(n_timeFilters, n_timeFilters*n_msFilters, (n_channs, msFilter_timeLen), dilation=(1,self.dilation_array[3]), groups=n_timeFilters)

        n_msFilters_total = n_timeFilters * n_msFilters * 4

        # Attention
        self.seg_att = seg_att               #  *2 等比缩放
        self.att_conv = nn.Conv2d(n_msFilters_total, n_msFilters_total, (1, self.seg_att), groups=n_msFilters_total)
        self.att_pool = nn.AvgPool2d((1, self.seg_att), stride=1)
        self.att_pointConv = nn.Conv2d(n_msFilters_total, n_msFilters_total, (1, 1))

        # projector avepooling+timeSmooth
        self.avgpool = nn.AvgPool2d((1, avgPoolLen))
        self.timeConv1 = nn.Conv2d(n_msFilters_total, n_msFilters_total * multiFact, (1, timeSmootherLen), groups=n_msFilters_total)
        self.timeConv2 = nn.Conv2d(n_msFilters_total * multiFact, n_msFilters_total * multiFact * multiFact, (1, timeSmootherLen), groups=n_msFilters_total * multiFact)
        # # pooling  时间上的max pooling目前不需要，因为最后输出层特征会整体做个时间上的平均,时间上用ave比max更符合直觉
        # self.maxPoolLen = maxPoolLen
        # self.maxpool = nn.MaxPool2d((1, self.maxPoolLen),self.maxPoolLen)
        # # self.flatten = nn.Flatten()
        # self.encoder = MLLA_BasicLayer(
        #                             in_dim=16, hidden_dim=64, out_dim=16,
        #                             depth=1, num_heads=2)
    
    def forward(self, input):
        # input.shape should be [B, dim, n_channel, T]
        if 'initial' in self.stratified:
            input = stratified_layerNorm(input, int(input.shape[0]/2))
        out = input
        # 1.1 original time conv
        # out = self.timeConv(input)
        # # # print(out.shape)

        # 1.2 MLLA channelwise
        # x = input
        # B, D1, n_channels, T = x.shape
        # x = x.permute(0, 2, 1, 3)  # Shape: [Batch, n_channels, D1, T]
        # x = x.reshape(B * n_channels, D1, T)  # Shape: [Batch * n_channels, D1, T]
        # x = x.permute(0, 2, 1)  # Shape: [Batch * n_channels, T, D1]
        # x = to_patch(x, patch_size=16, stride=2)  # Shape: [Batch * n_channels, N, D1]
        # x = self.encoder(x)  # Shape: [Batch * n_channels, N, out_dim]
        # x = x.view(B, n_channels, x.shape[1], x.shape[2])  # Shape: [Batch, n_channels, N, out_dim]
        # out = x.permute(0, 3, 1, 2)  # Shape: [Batch, out_dim, n_channels, N]
        # # print(x.shape, '\n')

        p = self.dilation_array * (self.msFilter_timeLen - 1)
        out1 = self.msConv1(F.pad(out, (int(p[0]//2), p[0]-int(p[0]//2)), "constant", 0))
        out2 = self.msConv2(F.pad(out, (int(p[1]//2), p[1]-int(p[1]//2)), "constant", 0))
        out3 = self.msConv3(F.pad(out, (int(p[2]//2), p[2]-int(p[2]//2)), "constant", 0))
        out4 = self.msConv4(F.pad(out, (int(p[3]//2), p[3]-int(p[3]//2)), "constant", 0))
        # print(out1.shape, out2.shape, out3.shape, out4.shape)
        out = torch.cat((out1, out2, out3, out4), 1) # (B, dims, 1, T)

        # Attention
        if self.has_att:
            att_w = F.relu(self.att_conv(F.pad(out, (self.seg_att-1, 0), "constant", 0)))
            if self.global_att:
                att_w = torch.mean(F.pad(att_w, (self.seg_att-1, 0), "constant", 0),-1).unsqueeze(-1) # (B, dims, 1, 1)
            else:
                att_w = self.att_pool(F.pad(att_w, (self.seg_att-1, 0), "constant", 0)) # (B, dims, 1, T)
            att_w = self.att_pointConv(att_w)
            if self.activ == 'relu':
                att_w = F.relu(att_w)
            elif self.activ == 'softmax':
                att_w = F.softmax(att_w / self.temp, dim=1)
            out = att_w * F.relu(out)          # (B, dims, 1, T)
        else:
            if self.extract_mode == 'me':
                out = F.relu(out)
        if self.saveFea:
            return out
        else:         # projecter
            if self.extract_mode == 'de':
                out = F.relu(out)
            out = self.avgpool(out)    # B*(t_dim*n_msFilters*4)*1*t_pool
            if 'middle1' in self.stratified:
                out = stratified_layerNorm(out, int(out.shape[0]/2))
            out = F.relu(self.timeConv1(out))
            out = F.relu(self.timeConv2(out))          #B*(t_dim*n_msFilters*4*multiFact*multiFact)*1*t_pool
            if 'middle2' in self.stratified:
                out = stratified_layerNorm(out, int(out.shape[0]/2))     
            proj_out = out.reshape(out.shape[0], -1)
            return F.normalize(proj_out, dim=1)

    
    def set_saveFea(self, saveFea):
        self.saveFea = saveFea

    def set_stratified(self,stratified):
        self.stratified = stratified

class MLLA_encoder(nn.Module):
    def __init__(self, in_dim=16, hid_dim=64, out_dim=16):
        super().__init__()
        self.encoder = MLLA_BasicLayer(
                                    in_dim=in_dim, hidden_dim=hid_dim, out_dim=out_dim,
                                    depth=1, num_heads=2)
        self.in_dim = in_dim
    
    def forward(self, input):
        # input.shape should be [B, dim, n_channel, T]
        x = stratified_layerNorm(input, int(input.shape[0]/2))
        B, D1, n_channels, T = x.shape
        x = x.permute(0, 2, 1, 3)  # Shape: [Batch, n_channels, D1, T]
        x = x.reshape(B * n_channels, D1, T)  # Shape: [Batch * n_channels, D1, T]
        x = x.permute(0, 2, 1)  # Shape: [Batch * n_channels, T, D1]
        x = to_patch(x, patch_size=self.in_dim, stride=2)  # Shape: [Batch * n_channels, N, D1]
        x = self.encoder(x)  # Shape: [Batch * n_channels, N, out_dim]
        x = x.view(B, n_channels, x.shape[1], x.shape[2])  # Shape: [Batch, n_channels, N, out_dim]
        return x
        # # print(x.shape, '\n')