import torch
import pytorch_lightning as pl
import torch.nn as nn
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from MLLA_test import MLLA_BasicLayer
# os.environ["GEOMSTATS_BACKEND"] = 'pytorch' 
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import matplotlib.pyplot as plt
import itertools
import time
import random
from utils_new import stratified_layerNorm, LDS_new
from model.models import Conv_att_simple_new
from model.loss.con_loss import SimCLRLoss
from modules import DimAttentionModule, Clisa_Proj, Channel_mlp, Replace_Encoder, MLLA_encoder

def logm(t):
    u, s, v = torch.svd(t) 
    return u @ torch.diag_embed(torch.log(s)) @ v.transpose(-2, -1)

def div_std(x):
    x = x.clone() / torch.std(x.clone())
    return x

def mean_to_zero(x):
    x = x.clone() - torch.mean(x.clone())
    return x

# def log_euclidean_normalization(A):
#     save_img(A.detach().cpu(), 'A_BEFORE.png')
#     A = (A + A.T) / 2
#     A_log = logm(A)
#     A_log_normalized = (A_log - A_log.mean()) / A_log.std()
#     A_normalized = torch.linalg.matrix_exp(A_log_normalized)
#     save_img(A_normalized.detach().cpu(), 'A_AFTER.png')
#     return A_normalized

def save_img(data, filename='image_with_colorbar.png', cmap='viridis'):
    # return
    print('called')
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    fig, ax = plt.subplots()
    cax = ax.imshow(data, cmap=cmap)
    fig.colorbar(cax)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def save_batch_images(data, folder='heatmaps', cmap='viridis'):
    # 创建保存热力图的文件夹
    if not os.path.exists(folder):
        os.makedirs(folder)

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    # 假设数据的形状为 [B, W, D]
    B, W, D = data.shape

    # 循环遍历每个批次，生成热力图
    for i in range(B):
        filename = os.path.join(folder, f'heatmap_{i}.png')
        fig, ax = plt.subplots()
        cax = ax.imshow(data[i], cmap=cmap)
        fig.colorbar(cax)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

def top_k_accuracy(logits, labels, ks=[1, 5]):
    """
    计算给定 logits 和 labels 的 top-k 准确率。
    
    参数：
    - logits: Tensor of shape (N, num_classes), 模型的预测输出
    - labels: Tensor of shape (N,), 包含实际类别的下标格式的标签
    - ks: List of integers, 指定要计算的 k 值，如 [1, 5] 表示计算 top-1 和 top-5 准确率
    
    返回：
    - acc_list: List of accuracies corresponding to each k in ks
    """
    max_k = max(ks)  # 计算需要的最大 k 值
    batch_size = labels.size(0)

    # 获取 top-k 预测，返回的 top_preds 形状为 (N, max_k)
    _, top_preds = logits.topk(max_k, dim=1, largest=True, sorted=True)

    # 扩展 labels 的形状以便与 top_preds 比较，形状为 (N, max_k)
    top_preds = top_preds.t()  # 转置为 (max_k, N)
    correct = top_preds.eq(labels.view(1, -1).expand_as(top_preds))  # 检查 top-k 内是否有正确的标签

    # 计算每个 k 对应的准确率
    acc_list = []
    for k in ks:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # 获取前 k 的正确预测数量
        accuracy = correct_k.mul_(100.0 / batch_size).item()  # 计算准确率百分比
        acc_list.append(accuracy)

    return acc_list

def is_spd(mat, tol=1e-6):
    return True
    for i in range(mat.shape[0]):
        for j in range(mat.shape[0]):
            if(mat[i][j] != mat[j][i]):
                print(mat[i][j], mat[j][i])
    return bool((mat == mat.T).all() and (torch.linalg.eigvals(mat).real>=0).all())

def compute_de(data):
    batch, n_t, n_channel = data.shape
    
    de_values = torch.zeros((batch, n_channel))
    
    for b in range(batch):
        for c in range(n_channel):
            channel_data = data[b, :, c]
            variance = torch.var(channel_data)
            de = 0.5 * torch.log(2 * torch.pi * torch.e * (variance + 1e-10))
            de_values[b, c] = de
    # de_values = de_values.reshape((batch, -1, n_channel, 1))
    de_values = stratified_layerNorm(de_values, n_samples=batch/2)
    # de_values = de_values.reshape((batch, -1))
    return de_values

class channelwiseEncoder(nn.Module):
    def __init__(self, standard_channels, n_filter):
        super().__init__()
        self.standard_channels = standard_channels
        self.n_channels = len(standard_channels)
        self.conv1d = nn.ModuleList([nn.Conv1d(1, n_filter, 8, 4, 1) for _ in range(self.n_channels)])

    def forward(self, x, x_channel_names):
        n_channels_data = len(x_channel_names)
        # assert n_channels_data == len(x_channel_names), "data channel not equal to channel_name dict length"
        outputs = []

        for i in range(n_channels_data):
            channel_data = x[:, :, i, :]
            # print(channel_data.shape)
            encoder_index = self.standard_channels.index(x_channel_names[i])
            out_channel_i = self.conv1d[encoder_index](channel_data)
            outputs.append(out_channel_i)
        outputs = torch.stack(outputs, dim=1)
        outputs = outputs.reshape((outputs.shape[0], -1, outputs.shape[-1]))
        # [batch, channel*n_filter, time]
        return outputs

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, dropout=0.1, bn='no'):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # if (bn == 'bn1') or (bn == 'bn2'):
        self.bn1 = nn.BatchNorm1d(hidden_dim, affine=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        # if bn == 'bn2':
        self.bn2 = nn.BatchNorm1d(hidden_dim//2, affine=False)
        self.fc3 = nn.Linear(hidden_dim//2, out_dim)
        self.bn = bn
        self.drop = nn.Dropout(p=dropout)
        # self.flag = False
    def forward(self, input):

        out = F.relu(self.fc1(input))

        if (self.bn == 'bn1') or (self.bn == 'bn2'):
            out = self.bn1(out)
        out = self.drop(out)
        out = F.relu(self.fc2(out))

        if self.bn == 'bn2':
            out = self.bn2(out)
        out = self.drop(out)
        out = self.fc3(out)

        return out

class BiMap(nn.Module):
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_shape, out_shape))
        # self.norm = nn.LayerNorm([out_shape, out_shape], elementwise_affine=False)
    def forward(self, x):
        out = self.W @ x @ self.W.transpose(-2, -1)
        out = (out - torch.mean(out, dim=0)) / torch.std(out, dim=0)
        # out = self.norm(out)
        return out
    
class ReEig(nn.Module):
    def __init__(self, eta):
        super().__init__()
        self.thresh = nn.Threshold(eta, eta)
    def forward(self, x):
        out = torch.zeros_like(x)
        for i, x_i in enumerate(x):
            eigvals, eigvecs = torch.linalg.eigh(x_i)
            eigvals = torch.diag(self.thresh(eigvals))
            out[i] = eigvecs @ eigvals @ eigvecs.transpose(-2, -1)
        return out
    
class ConvOut(nn.Module):
    # Input: a batch of SPD Covariance matrix [B, N, N]
    def __init__(self, in_shape):
        super().__init__()
        self.ReEig = ReEig(eta=0.005)
        self.bimap_1 = BiMap(in_shape=in_shape, out_shape=in_shape)
        self.bimap_2 = BiMap(in_shape=in_shape, out_shape=in_shape)

    def forward(self, x):
        x = self.bimap_1(x)
        x = self.ReEig(x)
        # x = self.bimap_2(x)
        # x = self.ReEig(x)
        out = x
        return out

class ConvOut_Euclidean(nn.Module):
    def __init__(self, out_channels=64, n_kernel=1):
        """
        SPD Convolution Layer with a pyramidal form.

        Args:
            out_channels (int): Number of output channels after each convolution layer.
            n_kernel (int): Number of convolution kernels (default is 1).
        """
        super(ConvOut_Euclidean, self).__init__()
        self.n_kernel = n_kernel
        self.out_channels = out_channels

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.ReLU()

        # Second convolutional layer with downsampling
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * 2,
            kernel_size=3,
            stride=2,  # Downsampling by reducing the spatial size
            padding=1,
            bias=False
        )
        self.norm2 = nn.BatchNorm2d(out_channels * 2)
        self.activation2 = nn.ReLU()

        # Third convolutional layer with further downsampling
        self.conv3 = nn.Conv2d(
            in_channels=out_channels * 2,
            out_channels=out_channels * 4,
            kernel_size=3,
            stride=2,  # Further downsampling
            padding=1,
            bias=False
        )
        self.norm3 = nn.BatchNorm2d(out_channels * 4)
        self.activation3 = nn.ReLU()

        # Linear layer to merge channels into a single channel via a linear combination
        self.merge = nn.Linear(out_channels * 4, 1, bias=False)

    def forward(self, x):
        """
        Forward pass of SPD Convolution

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, N], where:
                              B - Batch size
                              N - SPD matrix dimension

        Returns:
            torch.Tensor: Output tensor of shape [B, final_N, final_N] after merging channels.
        """
        B, N, _ = x.shape

        # Unsqueeze to add channel dimension: [B, 1, N, N]
        x = x.unsqueeze(1)

        # First convolution layer
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)

        # Second convolution layer with downsampling
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation2(x)

        # Third convolution layer with further downsampling
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activation3(x)

        # Permute to [B, final_N, final_N, out_channels * 4] to apply linear combination along channel dimension
        x = x.permute(0, 2, 3, 1)

        # Apply the linear layer to merge channels back to a single output [B, final_N, final_N, 1]
        x = self.merge(x)

        # Squeeze the last dimension to get the shape [B, final_N, final_N]
        x = x.squeeze(-1)

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

class   channel_MLLA(nn.Module):
    def __init__(self, patch_size, hidden_dim, out_dim, depth, patch_stride, drop_path, n_filter, filterLen, n_heads):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.encoders = nn.ModuleList([MLLA_BasicLayer(
        #                             in_dim=patch_size, hidden_dim=hidden_dim, out_dim=out_dim,
        #                             depth=1, num_heads=8) 
        #                             for _ in range(self.n_channels)])
        self.encoder = MLLA_BasicLayer(
                                    in_dim=patch_size, hidden_dim=hidden_dim, out_dim=out_dim,
                                    depth=depth, num_heads=n_heads)

    def forward(self, x):
        # print(x.shape)
        # x has shape [Batch, D1, n_channels, T]
        B, D1, n_channels, T = x.shape

        # Permute and reshape to combine batch and channel dimensions
        x = x.permute(0, 2, 1, 3)  # Shape: [Batch, n_channels, D1, T]
        x = x.reshape(B * n_channels, D1, T)  # Shape: [Batch * n_channels, D1, T]

        # Permute to match expected input shape for to_patch
        x = x.permute(0, 2, 1)  # Shape: [Batch * n_channels, T, D1]
        # print(x.shape)

        # Apply to_patch to all channels at once
        x = to_patch(x, patch_size=self.patch_size, stride=self.patch_stride)  # Shape: [Batch * n_channels, N, D1]

        # Pass through the encoder
        x = self.encoder(x)  # Shape: [Batch * n_channels, N, out_dim]

        # Reshape back to separate batch and channel dimensions
        x = x.view(B, n_channels, x.shape[1], x.shape[2])  # Shape: [Batch, n_channels, N, out_dim]

        # Permute dimensions as required
        x = x.permute(0, 1, 3, 2)  # Shape: [Batch, n_channels, out_dim, N]

        # Apply stratified_layerNorm
        x = stratified_layerNorm(x, n_samples=B // 2)

        # Permute back to original dimension order
        x = x.permute(0, 1, 3, 2)  # Shape: [Batch, n_channels, N, out_dim]

        return x


class Channel_Alignment(nn.Module):
    def __init__(self, source_channel, target_channel):
        super().__init__()
        # self.A = nn.Parameter(torch.eye(source_channel) if source_channel ==  target_channel else torch.randn(source_channel, target_channel))
        # self.A = nn.Parameter(torch.randn(source_channel, target_channel))
        self.fc1 = nn.Linear(source_channel, source_channel)
        self.fc2 = nn.Linear(source_channel, target_channel)
        # 30 * 60

    def forward(self, input):
        # [B, C, T, out_dim]
        out = input.permute(0, 2, 3, 1)
        # [B, T, out_dim, C]
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = out.permute(0, 3, 2, 1)
        [B, C, out_dim, T] = out.shape
        return out
    
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




class MultiModel_PL(pl.LightningModule):
    def __init__(self, cfg=None) -> None:
        super().__init__()
        self.cfg = cfg
        if self.cfg.channel_encoder.model == 'MLLA':
            self.channelwiseEncoder = channel_MLLA(
                patch_size=cfg.channel_encoder.patch_size,
                hidden_dim=cfg.channel_encoder.hidden_dim,
                out_dim=cfg.channel_encoder.out_dim,
                depth=cfg.channel_encoder.depth,
                patch_stride=cfg.channel_encoder.patch_stride,
                drop_path=cfg.channel_encoder.drop_path,
                n_filter=cfg.channel_encoder.n_filter,
                filterLen=cfg.channel_encoder.filterLen,
                n_heads=cfg.channel_encoder.n_heads)
            # self.channelwiseEncoder = channelwiseEncoder(n_filter=cfg.channel_encoder.out_dim, standard_channels=cfg.data_1.channels)
            self.alignmentModule_1 = Channel_Alignment(cfg.data_1.n_channs, cfg.align.n_channel_uni)
            self.alignmentModule_2 = Channel_Alignment(cfg.data_2.n_channs, cfg.align.n_channel_uni)
            self.alignmentModule_3 = Channel_Alignment(cfg.data_val.n_channs, cfg.align.n_channel_uni)
            self.sFilter = sFilter(dim_in=cfg.channel_encoder.out_dim, n_channs=cfg.align.n_channel_uni, sFilter_timeLen=3, multiFact=1)
            self.proj = Clisa_Proj(n_dim_in=cfg.channel_encoder.out_dim * cfg.align.n_channel_uni)
            # self.proj = Clisa_Proj(n_dim_in=cfg.align.n_channel_uni * cfg.channel_encoder.out_dim)
        
        if self.cfg.channel_encoder.model == 'replace':
            self.mlla = MLLA_encoder(in_dim=cfg.channel_encoder.patch_size,
                                    hid_dim=cfg.channel_encoder.hidden_dim,
                                    out_dim=cfg.channel_encoder.out_dim,)
            self.channelwiseEncoder = Replace_Encoder(n_timeFilters=16,
                                                    timeFilterLen=30,
                                                    n_msFilters=4,
                                                    msFilter_timeLen=3,
                                                    n_channs=cfg.align.n_channel_uni,
                                                    dilation_array=[1,3,6,12],
                                                    seg_att=15,
                                                    avgPoolLen=15,
                                                    timeSmootherLen=3,
                                                    multiFact=1,
                                                    stratified=['initial', 'middle1', 'middle2'],
                                                    activ='softmax',
                                                    temp=1.0,
                                                    saveFea=False,
                                                    has_att=True,
                                                    global_att=False)
            self.proj = Clisa_Proj(n_dim_in=256)
            self.c_mlp_0 = Channel_mlp(cfg.data_0.n_channs, cfg.align.n_channel_uni)
            self.c_mlp_1 = Channel_mlp(cfg.data_1.n_channs, cfg.align.n_channel_uni)
            self.c_mlp_2 = Channel_mlp(cfg.data_2.n_channs, cfg.align.n_channel_uni)
            self.c_mlp_3 = Channel_mlp(cfg.data_val.n_channs, cfg.align.n_channel_uni)
        # self.decoder = ConvOut(in_shape=cfg.align.n_channel_uni)
        # self.decoder = ConvOut_Euclidean(out_channels=64)
        self.proto = None
        # self.protos_1 = self.init_proto_rand(dim=cfg.align.n_channel_uni, n_class=cfg.data_1.n_class)
        # self.protos_2 = self.init_proto_rand(dim=cfg.align.n_channel_uni, n_class=cfg.data_2.n_class)
        self.lr = cfg.train.lr
        self.wd = cfg.train.wd
        self.max_epochs = cfg.train.max_epochs
        self.restart_times = cfg.train.restart_times
        self.is_logger = cfg.log.is_logger
        self.align_factor = cfg.align.factor
        # self.MLP_1 = MLP(input_dim=180, hidden_dim=64, out_dim=cfg.data_1.n_class)
        # self.MLP_2 = MLP(input_dim=180, hidden_dim=64, out_dim=cfg.data_2.n_class)
        # self.MLP_3 = MLP(input_dim=180, hidden_dim=64, out_dim=cfg.data_val.n_class)
        # self.dm = dm
        # self.train_dataset = dm.train_dataset
        # self.train_set_1 = self.train_dataset.dataset_a
        # self.train_set_2 = self.train_dataset.dataset_b
        self.criterion = SimCLRLoss(cfg.train.loss_temp)
        self.saveFea = False
        self.cov_feature_dim = cfg.align.n_channel_uni * cfg.channel_encoder.n_filter
        # self.cov_0_mean = nn.Parameter(torch.zeros([self.cov_feature_dim, self.cov_feature_dim]), requires_grad=False)
        # self.cov_1_mean = nn.Parameter(torch.zeros([self.cov_feature_dim, self.cov_feature_dim]), requires_grad=False)
        # self.cov_2_mean = nn.Parameter(torch.zeros([self.cov_feature_dim, self.cov_feature_dim]), requires_grad=False)
        self.phase = 'train'

    def cov_mat(self, data):
        # 计算每个dim维度下的通道间协方差矩阵，使用批量矩阵乘法。
        # 参数：
        #     data (Tensor): 输入数据，形状为 [B, dim, C, T]。
        # 返回：
        #     cov_matrix (Tensor): 每个dim的协方差矩阵，形状为 [B, dim, C, C]。
        B, dim, C, T = data.shape
        
        # 计算均值，形状为 [B, dim, C, 1]
        mean_data = data.mean(dim=-1, keepdim=True)  # 计算每个通道在T维度上的均值
        data_centered = data - mean_data  # 去中心化，形状 [B, dim, C, T]
        
        # 转置数据，形状变为 [B, dim, T, C]
        data_centered = data_centered.permute(0, 1, 3, 2)  # shape: [B, dim, T, C]
        
        # 将数据reshape成 [B * dim, C, T] 和 [B * dim, T, C]
        data_centered = data_centered.reshape(B * dim, C, T)  # shape: [B * dim, C, T]
        
        # 使用批量矩阵乘法计算协方差矩阵，结果形状为 [B * dim, C, C]
        cov_matrix = torch.bmm(data_centered, data_centered.permute(0, 2, 1)) / (T - 1)  # shape: [B * dim, C, C]
        # save_batch_images(cov_matrix, 'cov_matrices')
        
        # 将协方差矩阵重新reshape为 [B, dim, C, C]
        cov_matrix = cov_matrix.reshape(B, dim, C, C)  # shape: [B, dim, C, C]
        # save_batch_images(data[:,:200,:], 'cov_use_data')
        return cov_matrix

    def frechet_mean(self, cov_matrices, tol=1e-5, max_iter=100):
        # Ensure the input is a GPU tensor
        cov_matrices = cov_matrices.to('cuda') if not cov_matrices.is_cuda else cov_matrices
        
        centroid = torch.mean(cov_matrices, dim=0).to('cuda')
        for i in range(max_iter):
            prev_centroid = centroid.clone()
            
            # Compute weights in a vectorized manner
            distances = torch.linalg.norm(cov_matrices - centroid, dim=(1, 2), ord='fro') + 1e-8
            weights = 1.0 / distances
            weights /= torch.sum(weights)
            
            # Update centroid using the weights, avoiding explicit loops
            centroid = torch.einsum('i,ijk->jk', weights, cov_matrices)
            
            # Check for convergence
            delta = torch.linalg.norm(centroid - prev_centroid, ord='fro')
            if delta < tol:
                break
        
        # # Optional: Print optimized distance
        # optimized_dis = torch.linalg.norm(torch.mean(cov_matrices, dim=0) - centroid, ord='fro')
        # print(f'Optimized dis: {optimized_dis}')
        
        return centroid

    # CDA for Cross_dataset Alignment
    def CDA_loss(self, cov_0, cov_1, cov_2):
        cov_0 = cov_0.reshape(2, -1, self.cfg.channel_encoder.out_dim, self.cfg.align.n_channel_uni, self.cfg.align.n_channel_uni)
        cov_1 = cov_1.reshape(2, -1, self.cfg.channel_encoder.out_dim, self.cfg.align.n_channel_uni, self.cfg.align.n_channel_uni)
        cov_2 = cov_2.reshape(2, -1, self.cfg.channel_encoder.out_dim, self.cfg.align.n_channel_uni, self.cfg.align.n_channel_uni)
        # print(cov_0.shape, cov_1.shape, cov_2.shape)
        # cen_0 = torch.zeros(self.cfg.channel_encoder.out_dim, self.cfg.align.n_channel_uni, self.cfg.align.n_channel_uni)
        # cen_1 = torch.zeros(self.cfg.channel_encoder.out_dim, self.cfg.align.n_channel_uni, self.cfg.align.n_channel_uni)
        # cen_2 = torch.zeros(self.cfg.channel_encoder.out_dim, self.cfg.align.n_channel_uni, self.cfg.align.n_channel_uni)
        dis = 0
        for dim in range(self.cfg.channel_encoder.out_dim):
            ind_cen = []
            ind_cen.append(self.get_ind_cen(cov_0[0][:, dim]))
            ind_cen.append(self.get_ind_cen(cov_0[1][:, dim]))
            ind_cen.append(self.get_ind_cen(cov_1[0][:, dim]))
            ind_cen.append(self.get_ind_cen(cov_1[1][:, dim]))
            ind_cen.append(self.get_ind_cen(cov_2[0][:, dim]))
            ind_cen.append(self.get_ind_cen(cov_2[1][:, dim]))
            for i in range(len(ind_cen)):
                for j in range(i+1, len(ind_cen)):
                    dis = dis + self.frobenius_distance(ind_cen[i], ind_cen[j])
        # save_batch_images(cen_0, 'cen_0_sample')
        loss = torch.log(dis + 1.0)
        # save_img(torch.concat([cen_0, cen_1, cen_2]), 'cov_cen.png')
        return loss, None, None, None

    def frobenius_distance(self, matrix_a, matrix_b):
        return torch.linalg.norm(matrix_a-matrix_b, 'fro')
        # 计算Frobenius距离
        if matrix_a.shape != matrix_b.shape:
            raise ValueError("Must have same shape")
        inner_term = matrix_a - matrix_b
        inner_multi = inner_term @ inner_term.transpose(-1, -2)
        _, s, _= torch.svd(inner_multi) 
        distance = torch.sum(s, dim=-1)
        return distance

    def top1_accuracy(self, dis_mat, labels):
        # save_img(dis_mat, 'MLP_logits.png')
        label_onehot = torch.nn.functional.one_hot(labels, num_classes=9)
        # save_img(label_onehot, 'label_MLP.png')
        # Get the predicted classes (indices of the 到原型最小距离 in each row)
        predicted_classes = torch.argmax(dis_mat, dim=1)
        
        # Flatten labels to match predicted_classes shape
        labels = labels.view(-1)
        
        # Calculate the number of correct predictions
        correct_predictions = (predicted_classes == labels).sum().item()
        
        # Calculate accuracy
        accuracy = correct_predictions / labels.size(0)
        
        return accuracy

    def init_proto_rand(self, dim, n_class):
        T_init = 64
        prototype_init = torch.zeros((n_class, dim, dim))
        for i in range(n_class):
            prototype_init[i] = div_std(self.cov_mat(torch.randn(1, T_init, dim)))
        return nn.Parameter(prototype_init)

    # def loss_clisa(self, cov, emo_label):
    #     if self.cfg.align.to_riem:
    #         cov = logm(cov)
    #     N, C, _ = cov.shape
    #     n_vid = cov.shape[0] // 2
    #     mat_a = cov.unsqueeze(0).repeat(N, 1, 1, 1)  # [N, N, C, C]
    #     mat_b = cov.unsqueeze(1).repeat(1, N, 1, 1)  # [N, N, C, C]
    #     similarity_matrix = -self.frobenius_distance(mat_a, mat_b)
        
    #     # save_img(similarity_matrix, 'dis_pair.png')
    #     # similarity_matrix = torch.zeros((cov.shape[0], cov.shape[0]))
    #     labels = torch.cat([torch.arange(n_vid) for i in range(2)], dim=0)
    #     labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        
    #     # for i in range(cov.shape[0]):
    #     #     for j in range(i, cov.shape[0]):
    #     #         cov_i = cov[i]
    #     #         cov_j = cov[j]
    #     #         similarity_matrix[i][j] = -self.frobenius_distance(cov_i, cov_j)
    #     #         similarity_matrix[j][i] = similarity_matrix[i][j]

    #     mask = torch.eye(labels.shape[0], dtype=torch.bool)
    #     labels = labels[~mask].view(labels.shape[0], -1)
    #     similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    #     similarity_matrix = mean_to_zero(similarity_matrix)
    #     similarity_matrix = div_std(similarity_matrix)

    #     # save_img(similarity_matrix, 'dis_pair.png')
    #     # save_img(labels.detach().cpu(), 'label_pair.png')

    #     positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    #     negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    #     logits = torch.cat([negatives, positives], dim=1)
    #     # save_img(logits, 'logits_rearranged.png')
    #     labels = torch.ones(logits.shape[0], dtype=torch.long)*(logits.shape[1]-1)
    #     num_classes = labels.max().item() + 1
    #     # print(label)
    #     label_onehot = torch.nn.functional.one_hot(labels, num_classes=num_classes)
    #     # save_img(label_onehot, 'labels_rearranged.png')

    #     CEloss = nn.CrossEntropyLoss()
    #     pair_loss = CEloss(logits, labels)
    #     return pair_loss

    def restore_diagonal(self, matrix):
        n = matrix.shape[0]
        assert matrix.shape[1] == n - 1, "Input must be of shape [n, n-1]"
        restored_matrix = torch.zeros(n, n)
        mask = ~torch.eye(n, dtype=torch.bool)
        restored_matrix[mask] = matrix.view(-1)
        return restored_matrix

    def loss_clisa_fea(self, fea, temperature=0.3):

        loss, logits, logits_labels = self.criterion(fea)
        [acc_1, acc_5] = top_k_accuracy(logits, logits_labels)
        return loss, acc_1, acc_5
    
        N, C = fea.shape
        n_vid = N // 2
        # save_img(fea, 'fea_debug.png')
        similarity_matrix = torch.matmul(fea, fea.T)
        # save_img(similarity_matrix, 'dis_pair.png')
        
        # similarity_matrix = torch.zeros((cov.shape[0], cov.shape[0]))
        labels = torch.cat([torch.arange(n_vid) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        
        # for i in range(cov.shape[0]):
        #     for j in range(i, cov.shape[0]):
        #         cov_i = cov[i]
        #         cov_j = cov[j]
        #         similarity_matrix[i][j] = -self.frobenius_distance(cov_i, cov_j)
        #         similarity_matrix[j][i] = similarity_matrix[i][j]

        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # similarity_matrix = mean_to_zero(similarity_matrix)
        # similarity_matrix = div_std(similarity_matrix)

        # save_img(labels.detach().cpu(), 'label_pair.png')

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        logits = torch.cat([negatives, positives], dim=1)
        logits = logits / temperature
        # save_img(logits, 'logits_rearranged.png')
        labels = torch.ones(logits.shape[0], dtype=torch.long)*(logits.shape[1]-1)
        num_classes = labels.max().item() + 1
        # print(label)
        label_onehot = torch.nn.functional.one_hot(labels, num_classes=num_classes)
        # save_img(label_onehot, 'labels_rearranged.png')
        CEloss = nn.CrossEntropyLoss()
        pair_loss = CEloss(logits, labels)
        [acc_1, acc_5] = top_k_accuracy(logits, labels)
        return pair_loss, acc_1, acc_5



    # def loss_proto(self, cov, label, protos):
    #     # print(cov.shape)
    #     # print(label.shape)
    #     batch_size = cov.shape[0]
    #     loss = 0
    #     dis_mat = torch.zeros((batch_size, protos.shape[0]))
    #     for i, cov_i in enumerate(cov):
    #         # save_img(cov_i.detach().cpu(), 'cov_i.png')
    #         for j, pro_j in enumerate(protos):
    #             # pro_j = div_std(pro_j)
    #             # print(is_spd(cov_i))
    #             # print(is_spd(pro_j))
    #             dis_mat[i][j] = -self.frobenius_distance(cov_i, pro_j)
    #             # save_img(pro_j.detach().cpu(), 'pro_j.png')
    #             # print(-dis_mat[i][j])
    #         dis_mat[i] = div_std(dis_mat[i])
    #         dis_mat[i] = mean_to_zero(dis_mat[i])
    #         dis_mat[i] = F.softmax(dis_mat[i])

        
    #     save_img(dis_mat.detach().cpu(), 'dis_pro.png')
    #     num_classes = label.max().item() + 1
    #     # print(label)
    #     label_onehot = torch.nn.functional.one_hot(label, num_classes=num_classes)
    #     save_img(label_onehot.detach().cpu(), 'label_pro.png')

    #     # print(dis_mat)
    #     acc = self.top1_accuracy(dis_mat, label)
    #     # dis_mat = F.log_softmax(dis_mat, dim=1)
    #     CEloss = nn.CrossEntropyLoss()
    #     # NLLLoss = nn.NLLLoss()
    #     # loss = NLLLoss(dis_mat, label)
    #     loss_ce = CEloss(dis_mat, label)
    #     # print(loss)
    #     # print(loss_ce)
    #     # print(loss)

    #     return loss_ce, acc

    # def loss_MLP(self, feature, labels, MLP_):
    #     # print(feature.shape)
    #     B, T, n_dim, n_channel = feature.shape
    #     feature = feature.permute(0, 2, 3, 1)
    #     feature = feature.reshape(B, n_dim*n_channel, T)
    #     # LDS Smooth
    #     save_img(feature[4], 'feature_before_lds.png')
    #     feature = feature.permute(0, 2, 1)
    #     feature = LDS_new(feature)
    #     feature = feature.permute(0, 2, 1)
    #     save_img(feature[4], 'feature_after_lds.png')
    #     # Ave Pooling
    #     feature = F.avg_pool1d(feature, kernel_size=T // 5)
    #     feature = F.avg_pool1d(feature, kernel_size=feature.shape[-1])
    #     # print(feature.shape)
    #     feature = feature.reshape(B, -1)
    #     save_img(feature, 'feature_lds.png')
    #     # feature = feature.detach()
    #     logits = MLP_(feature)
    #     CEloss = torch.nn.CrossEntropyLoss()
    #     loss = CEloss(logits, labels)
    #     save_img(logits, 'MLP_logits.png')
    #     acc = self.top1_accuracy(logits, labels)
    #     return loss, acc
    
    def extract_leading_eig_vector(self, data_dim, data, device='gpu'):
        geom_backend = torch if device == 'gpu' else np
        # data_dim = 4
        matrix_data = geom_backend.zeros([data.shape[0],data_dim,data_dim])
        
        for specimen in range(data.shape[0]):
            this_specimen_data = data[specimen,:]
            this_specimen_matrix = geom_backend.zeros([data_dim,data_dim])
            # for 
            # this_specimen_matrix[i,j]
            for row in range(data_dim):
                
                col_data = geom_backend.concatenate((geom_backend.zeros(row),this_specimen_data[int((row*(data_dim*2-row+1))/2):int(((row*(data_dim*2-row+1))/2))+data_dim-row]))
                
                this_specimen_matrix[row,:] = col_data
                
            this_specimen_matrix = (this_specimen_matrix - geom_backend.eye(data_dim,data_dim) * this_specimen_matrix).T +this_specimen_matrix
            
            matrix_data[specimen,:,:] = this_specimen_matrix
            
            # reshape_matrix_data
        return matrix_data   

    def get_ind_cen(self, mat):
        if not self.cfg.align.to_riem:
            return torch.mean(mat, dim=0)
        mat = torch.squeeze(mat)
        mat = self.frechet_mean(mat)
        return mat
    
    def forward(self, x, dataset='0', mode = 'dev'):
        if(mode == 'dev'):
            x = self.mlla(x)
            if dataset == '0':
                x = self.c_mlp_0(x)
            if dataset == '1':
                x = self.c_mlp_1(x)
            if dataset == '2':
                x = self.c_mlp_2(x)
            if dataset == '3':
                x = self.c_mlp_3(x)
            if self.saveFea:
                self.channelwiseEncoder.saveFea = True
            fea = self.channelwiseEncoder(x)
            if self.saveFea:
                return fea
            else:         
                return x, fea
        else:
            # fea = self.channelwiseEncoder(x)
            # if dataset == '1':
            #     out = self.alignmentModule_1(fea)
            # if dataset == '2':
            #     out = self.alignmentModule_2(fea)
            # if dataset == '3':
            #     out = self.alignmentModule_1(fea)
            # # [B, n_c, dim, T]
            # out = self.dim_att(out)
            # if self.saveFea:
            #     pred = self.sFilter(out).permute(0, 2, 1, 3)
            #     return pred
            # else:         # projecter
            #     return out, self.proj(out)
            pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        # scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.max_epochs, gamma=0.8, last_epoch=-1, verbose=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.max_epochs // self.restart_times, eta_min=0,last_epoch=-1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    def set_phase(self, phase):
        self.phase = phase
        if phase == 'train':
            self.lr = self.cfg.train.lr
            self.wd = self.cfg.train.wd    
        elif phase == 'finetune':
            self.lr = self.cfg.finetune.lr
            self.wd = self.cfg.finetune.wd
            self.proj.requires_grad_(False)
            self.channelwiseEncoder.requires_grad_(False)
            self.mlla.requires_grad_(False)
            # self.sFilter.requires_grad_(False)
        return
    # remain to be implemented
    def training_step(self, batch, batch_idx):
        if self.phase == 'train':
            [x_0, x_1, x_2, x_3], [y_0, y_1, y_2, y_3] = batch
            x_0 = x_0[0]
            x_1 = x_1[0]
            x_2 = x_2[0]
            x_3 = x_3[0]
            y_0 = y_0[0]
            y_1 = y_1[0]
            y_2 = y_2[0]
            y_3 = y_3[0]
            # x_1, y_1 = batch
            # print(x_1.shape, y_1.shape)

            fea_0, fea_clisa_0 = self.forward(x_0, '0')
            fea_1, fea_clisa_1 = self.forward(x_1, '1')
            fea_2, fea_clisa_2 = self.forward(x_2, '2')
            # fea_3, fea_clisa_3 = self.forward(x_3, '3')
            
            loss = 0
            loss_0 = 0
            loss_1 = 0
            loss_2 = 0

            # 1. proto_loss
            # if self.cfg.align.proto_loss:
            #     if self.proto is None:
            #         self.proto = self.init_proto_rand(dim=self.cfg.align.n_channel_uni, n_class=9)
            #     cov_1 = self.cov_mat(fea_1)
            #     cov_2 = self.cov_mat(fea_2)
            #     for i in range(len(cov_1)):
            #         cov_1[i] = div_std(cov_1[i])
            #     for i in range(len(cov_2)):
            #         cov_2[i] = div_std(cov_2[i])
            #     loss_proto_1, acc_1 = self.loss_proto(cov_1, y_1, self.proto)
            #     loss_proto_2, acc_2 = self.loss_proto(cov_2, y_2, self.proto)
            #     loss_1 = loss_1 + loss_proto_1
            #     loss_2 = loss_2 + loss_proto_2
            #     loss = loss + loss_proto_1 + loss_proto_2
            #     self.log_dict({
            #             'loss_proto_1/train': loss_proto_1, 
            #             'loss_proto_2/train': loss_proto_2, 
            #             'acc_proto_1/train': acc_1,
            #             'acc_proto_2/train': acc_2,
            #             },
            #             logger=self.is_logger,
            #             on_step=False, on_epoch=True, prog_bar=True)
            #     # print(f'loss_proto_1={loss_proto_1} loss_proto_2={loss_proto_2}\nacc_proto_1={acc_1} acc_proto_2={acc_2}')

            # 2. clisa_loss
            if self.cfg.align.clisa_loss:
                # loss_clisa_1 = self.loss_clisa(cov_1, y_1)
                # loss_clisa_2 = self.loss_clisa(cov_2, y_2)
                loss_clisa_0, acc1_0, acc5_0 = self.loss_clisa_fea(fea_clisa_0)
                loss_clisa_1, acc1_1, acc5_1 = self.loss_clisa_fea(fea_clisa_1)
                loss_clisa_2, acc1_2, acc5_2 = self.loss_clisa_fea(fea_clisa_2)
                loss_0 = loss_0 + loss_clisa_0
                loss_1 = loss_1 + loss_clisa_1
                loss_2 = loss_2 + loss_clisa_2
                loss = loss + loss_clisa_0
                loss = loss + loss_clisa_1
                loss = loss + loss_clisa_2
                self.log_dict({
                        'loss_clisa_0/train': loss_clisa_0, 
                        'loss_clisa_1/train': loss_clisa_1, 
                        'loss_clisa_2/train': loss_clisa_2, 
                        'acc1_0/train': acc1_0, 
                        'acc1_1/train': acc1_1, 
                        'acc1_2/train': acc1_2,
                        'acc5_0/train': acc5_0, 
                        'acc5_1/train': acc5_1, 
                        'acc5_2/train': acc5_2,
                        },
                        logger=self.is_logger,
                        on_step=False, on_epoch=True, prog_bar=True)
                # print(f'loss_clisa_1={loss_clisa_1} loss_clisa_2={loss_clisa_2} ')
            
            # 3. emotion MLP classification
            # if self.cfg.align.MLP_loss:
            #     loss_MLP_1, acc_MLP_1 = self.loss_MLP(fea_1, y_1, self.MLP_1)
            #     loss_MLP_2, acc_MLP_2 = self.loss_MLP(fea_2, y_2, self.MLP_2)
            #     self.log_dict({
            #         'loss_MLP_1/train': loss_MLP_1, 
            #         'loss_MLP_2/train': loss_MLP_2, 
            #         'loss_MLP_3/train': loss_MLP_3, 
            #         'acc_MLP_1/train': acc_MLP_1, 
            #         'acc_MLP_2/train': acc_MLP_2,  
            #         'acc_MLP_3/train': acc_MLP_3,      
            #         },
            #         logger=self.is_logger,
            #         on_step=False, on_epoch=True, prog_bar=True)
            #     loss = loss + loss_MLP_1 + loss_MLP_2
            #     loss = loss + loss_MLP_3

            
            # 4. cov Riemanian align loss
            loss_align = 0
            cov_0 = self.cov_mat(fea_0) # [B_0, dim, C, C]
            cov_1 = self.cov_mat(fea_1)
            cov_2 = self.cov_mat(fea_2)
            # cov_3 = self.cov_mat(fea_3)
            if self.cfg.align.align_loss:
                cen_loss, cen_0, cen_1, cen_2 = self.CDA_loss(cov_0, cov_1, cov_2)
            else:
                cen_loss, cen_0, cen_1, cen_2 = self.CDA_loss(cov_0.detach(), cov_1.detach(), cov_2.detach())
            # with torch.no_grad():
            #     self.cov_0_mean += cen_0 / self.cfg.train.n_pairs
            #     self.cov_1_mean += cen_1 / self.cfg.train.n_pairs
            #     self.cov_2_mean += cen_2 / self.cfg.train.n_pairs
            # save_batch_images(torch.concat([self.cov_0_mean, self.cov_1_mean, self.cov_2_mean]).unsqueeze(0), 'cov_mean')
            # save_batch_images(torch.concat([cen_0, cen_1, cen_2]).unsqueeze(0), 'cov_cen')
            loss_align = loss_align + cen_loss
            # loss_align = loss_align + self.CDA_loss(cov_1.detach(), cov_3)
            # loss_align = loss_align + self.CDA_loss(cov_2.detach(), cov_3)
            # print(f'loss_align={loss_align}')
            self.log_dict({
                'loss_align/train': loss_align,    
                },
                logger=self.is_logger,
                on_step=False, on_epoch=True, prog_bar=True)
            if self.cfg.align.align_loss:
                loss = loss + loss_align
                
            self.log_dict({
                    'loss_total/train': loss, 
                    },
                    logger=self.is_logger,
                    on_step=False, on_epoch=True, prog_bar=True)
        
        elif self.phase == 'finetune':
            # save_batch_images(torch.concat([self.cov_0_mean, self.cov_1_mean, self.cov_2_mean]).unsqueeze(0), 'cov_cen')
            loss_align = 0
            [x_3], [y_3] = batch
            x_3 = x_3[0]
            y_3 = y_3[0]
            fea_3, fea_clisa_3 = self.forward(x_3, '3')
            loss_clisa_3, acc1_3, acc5_3 = self.loss_clisa_fea(fea_clisa_3)
            loss = loss_clisa_3
            if(self.cfg.finetune.align):
                cov_3 = self.cov_mat(fea_3)
                cen_3 = self.get_ind_cen(cov_3)
                # save_batch_images(cov_3, 'cov_3')
                dis = 0
                dis = dis + self.frobenius_distance(cen_3, self.cov_0_mean)
                dis = dis + self.frobenius_distance(cen_3, self.cov_1_mean)
                dis = dis + self.frobenius_distance(cen_3, self.cov_2_mean)
                loss_align = dis
                loss_align = torch.log(dis + 1.0)
                loss = loss +loss_align
            self.log_dict({
                    'loss_align/train': loss_align, 
                    'loss_clisa/train': loss_clisa_3,   
                    'acc1_3/train': acc1_3,
                    'loss_total/train': loss, 
                    },
                    logger=self.is_logger,
                    on_step=False, on_epoch=True, prog_bar=True)
        

        # Check grad 
        check_grad = 0
        if check_grad:
            for name, param in self.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    print(f'grad_norm_{name}', grad_norm)

        return loss



    
    def validation_step(self, batch, batch_idx):
        if self.phase == 'train':
            [x_0, x_1, x_2, x_3], [y_0, y_1, y_2, y_3] = batch
            x_0 = x_0[0]
            x_1 = x_1[0]
            x_2 = x_2[0]
            x_3 = x_3[0]
            y_0 = y_0[0]
            y_1 = y_1[0]
            y_2 = y_2[0]
            y_3 = y_3[0]
            # x_1, y_1 = batch
            # print(x_1.shape, y_1.shape)

            fea_0, fea_clisa_0 = self.forward(x_0, '0')
            fea_1, fea_clisa_1 = self.forward(x_1, '1')
            fea_2, fea_clisa_2 = self.forward(x_2, '2')
            # fea_3, fea_clisa_3 = self.forward(x_3, '3')
            
            # cov_1 = torch.mean(self.cov_mat(fea_1), dim=0) / self.cfg.train.n_pairs
            # cov_2 = torch.mean(self.cov_mat(fea_2), dim=0) / self.cfg.train.n_pairs
            # with torch.no_grad():
            #     self.cov_1_mean += cov_1  # 使用 in-place 加法操作
            #     self.cov_2_mean += cov_2  # 使用 in-place 加法操作

            # save_batch_images(torch.stack([self.cov_1_mean, self.cov_2_mean]), 'cov_mean_extractor')


            loss = 0
            loss_0 = 0
            loss_1 = 0
            loss_2 = 0

            # 1. proto_loss
            if self.cfg.align.proto_loss:
                if self.proto is None:
                    self.proto = self.init_proto_rand(dim=self.cfg.align.n_channel_uni, n_class=9)
                cov_1 = self.cov_mat(fea_1)
                cov_2 = self.cov_mat(fea_2)
                for i in range(len(cov_1)):
                    cov_1[i] = div_std(cov_1[i])
                for i in range(len(cov_2)):
                    cov_2[i] = div_std(cov_2[i])
                loss_proto_1, acc_1 = self.loss_proto(cov_1, y_1, self.proto)
                loss_proto_2, acc_2 = self.loss_proto(cov_2, y_2, self.proto)
                loss_1 = loss_1 + loss_proto_1
                loss_2 = loss_2 + loss_proto_2
                loss = loss + loss_proto_1 + loss_proto_2
                self.log_dict({
                        'loss_proto_1/val': loss_proto_1, 
                        'loss_proto_2/val': loss_proto_2, 
                        'acc_proto_1/val': acc_1,
                        'acc_proto_2/val': acc_2,
                        },
                        logger=self.is_logger,
                        on_step=False, on_epoch=True, prog_bar=True)
                # print(f'loss_proto_1={loss_proto_1} loss_proto_2={loss_proto_2}\nacc_proto_1={acc_1} acc_proto_2={acc_2}')

            # 2. clisa_loss
            if self.cfg.align.clisa_loss:
                # loss_clisa_1 = self.loss_clisa(cov_1, y_1)
                # loss_clisa_2 = self.loss_clisa(cov_2, y_2)
                loss_clisa_0, acc1_0, acc5_0 = self.loss_clisa_fea(fea_clisa_0)
                loss_clisa_1, acc1_1, acc5_1 = self.loss_clisa_fea(fea_clisa_1)
                loss_clisa_2, acc1_2, acc5_2 = self.loss_clisa_fea(fea_clisa_2)
                loss_0 = loss_0 + loss_clisa_0
                loss_1 = loss_1 + loss_clisa_1
                loss_2 = loss_2 + loss_clisa_2
                loss = loss + loss_clisa_0
                loss = loss + loss_clisa_1
                loss = loss + loss_clisa_2
                self.log_dict({
                        'loss_clisa_0/val': loss_clisa_0, 
                        'loss_clisa_1/val': loss_clisa_1, 
                        'loss_clisa_2/val': loss_clisa_2, 
                        'acc1_0/val': acc1_0, 
                        'acc1_1/val': acc1_1, 
                        'acc1_2/val': acc1_2,
                        'acc5_0/val': acc5_0, 
                        'acc5_1/val': acc5_1, 
                        'acc5_2/val': acc5_2,
                        },
                        logger=self.is_logger,
                        on_step=False, on_epoch=True, prog_bar=True)
                # print(f'loss_clisa_1={loss_clisa_1} loss_clisa_2={loss_clisa_2} ')
            
            # 3. emotion MLP classification
            if self.cfg.align.MLP_loss:
                loss_MLP_1, acc_MLP_1 = self.loss_MLP(fea_1, y_1, self.MLP_1)
                loss_MLP_2, acc_MLP_2 = self.loss_MLP(fea_2, y_2, self.MLP_2)
                self.log_dict({
                    'loss_MLP_1/val': loss_MLP_1, 
                    'loss_MLP_2/val': loss_MLP_2, 
                    'loss_MLP_3/val': loss_MLP_3, 
                    'acc_MLP_1/val': acc_MLP_1, 
                    'acc_MLP_2/val': acc_MLP_2,  
                    'acc_MLP_3/val': acc_MLP_3,      
                    },
                    logger=self.is_logger,
                    on_step=False, on_epoch=True, prog_bar=True)
                loss = loss + loss_MLP_1 + loss_MLP_2
                loss = loss + loss_MLP_3

            self.log_dict({
                    'loss_total/val': loss, 
                    },
                    logger=self.is_logger,
                    on_step=False, on_epoch=True, prog_bar=True)
            
            # 4. cov Riemanian align loss
            loss_align = 0
            cov_0 = self.cov_mat(fea_0)
            cov_1 = self.cov_mat(fea_1)
            cov_2 = self.cov_mat(fea_2)
            # cov_3 = self.cov_mat(fea_3)
            if self.cfg.align.align_loss:
                cen_loss, cen_0, cen_1, cen_2 = self.CDA_loss(cov_0, cov_1, cov_2)
            else:
                cen_loss, cen_0, cen_1, cen_2 = self.CDA_loss(cov_0.detach(), cov_1.detach(), cov_2.detach())
            # with torch.no_grad():
            #     self.cov_0_mean += cen_0 / self.cfg.train.n_pairs
            #     self.cov_1_mean += cen_1 / self.cfg.train.n_pairs
            #     self.cov_2_mean += cen_2 / self.cfg.train.n_pairs
            # save_batch_images(torch.concat([self.cov_0_mean, self.cov_1_mean, self.cov_2_mean]).unsqueeze (0), 'cov_mean')
            # save_batch_images(torch.concat([cen_0, cen_1, cen_2]).unsqueeze(0), 'cov_cen')
            loss_align = loss_align + cen_loss
            # loss_align = loss_align + self.CDA_loss(cov_1.detach(), cov_3)
            # loss_align = loss_align + self.CDA_loss(cov_2.detach(), cov_3)
            # print(f'loss_align={loss_align}')
            self.log_dict({
                'loss_align/val': loss_align,    
                },
                logger=self.is_logger,
                on_step=False, on_epoch=True, prog_bar=True)
            if self.cfg.align.align_loss:
                loss = loss + loss_align

            # Check grad 
            check_grad = 0
            if check_grad:
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        if 'encoder' not in name:
                            grad_norm = param.grad.data.norm(2).item()
                            print(f'grad_norm_{name}', grad_norm)
                        else:
                            grad_norm = param.grad.data.norm(2).item()
                            print(f'grad_norm_{name}', grad_norm)
                            pass
            return loss
        
        elif self.phase == 'finetune':
            loss_align = 0
            [x_3], [y_3] = batch
            x_3 = x_3[0]
            y_3 = y_3[0]
            fea_3, fea_clisa_3 = self.forward(x_3, '3')
            loss_clisa_3, acc1_3, acc5_3 = self.loss_clisa_fea(fea_clisa_3)
            loss = loss_clisa_3
            if(self.cfg.finetune.align):
                cov_3 = self.cov_mat(fea_3)
                cen_3 = self.get_ind_cen(cov_3)
                dis = 0
                dis = dis + self.frobenius_distance(cen_3, self.cov_0_mean)
                dis = dis + self.frobenius_distance(cen_3, self.cov_1_mean)
                dis = dis + self.frobenius_distance(cen_3, self.cov_2_mean)
                loss_align = dis
                loss_align = torch.log(dis + 1.0)
                loss = loss +loss_align
            self.log_dict({
                    'loss_align/val': loss_align, 
                    'loss_clisa/val': loss_clisa_3,   
                    'acc1_3/val': acc1_3,
                    'loss_total/val': loss, 
                    },
                    logger=self.is_logger,
                    on_step=False, on_epoch=True, prog_bar=True)
            return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        # fea_1 = self.cnn_encoder(x_1)
        # fea_2 = self.cnn_encoder(x_2)
        # fea_3 = self.cnn_encoder(x_3)

        fea = self.forward(x, dataset='3')
        return fea
    