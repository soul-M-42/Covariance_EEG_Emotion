import torch
import pytorch_lightning as pl
import torch.nn as nn
from model.models import Channel_Alignment
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from MLLA_test import MLLA_BasicLayer

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

def to_patch(data, patch_size=50, stride=25):
    batchsize, timelen, dim = data.shape
    num_patches = (timelen - patch_size) // stride + 1
    patches = torch.zeros((batchsize, num_patches, patch_size)).to('cuda')
    for i in range(num_patches):
        start_idx = i * stride
        end_idx = start_idx + patch_size
        patches[:, i, :] = data[:, start_idx:end_idx, 0]
    return patches

class channel_MLLA(nn.Module):
    def __init__(self, standard_channels, patch_size, hidden_dim, out_dim, patch_stride):
        super().__init__()
        self.standard_channels = standard_channels
        self.n_channels = len(standard_channels)
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.encoders = nn.ModuleList([MLLA_BasicLayer(
                                    in_dim=patch_size, hidden_dim=hidden_dim, out_dim=out_dim,
                                    depth=2, num_heads=8) 
                                    for _ in range(self.n_channels)])

    def forward(self, x, x_channel_names):
        n_channels_data = len(x_channel_names)
        # assert n_channels_data == len(x_channel_names), "data channel not equal to channel_name dict length"
        outputs = []

        for i in range(n_channels_data):
            channel_data = x[:, :, i, :]
            # print(channel_data.shape)
            # [Batch, channel, time]
            channel_data = channel_data.permute(0, 2, 1)
            channel_data = to_patch(channel_data, patch_size=self.patch_size, stride=self.patch_stride)
            # print(channel_data.shape)
            # [Batch, time(N), channel(C)]
            encoder_index = self.standard_channels.index(x_channel_names[i])
            out_channel_i = self.encoders[encoder_index](channel_data)
            channel_data = channel_data.permute(0, 2, 1)
            outputs.append(out_channel_i)
        outputs = torch.stack(outputs, dim=1)
        outputs = outputs.permute(0, 1, 3, 2)
        outputs = outputs.reshape((outputs.shape[0], -1, outputs.shape[-1]))
        # [batch, channel*n_filter, time]
        return outputs

class Channel_Alignment(nn.Module):
    def __init__(self, source_channel, target_channel):
        super().__init__()
        # self.A = nn.Parameter(torch.eye(source_channel) if source_channel ==  target_channel else torch.randn(source_channel, target_channel))
        self.A = nn.Parameter(torch.randn(source_channel, target_channel))
        # 30 * 60

    def forward(self, input):
        # input.shape = [batch, channel*n_filter, time']
        # 60(ch) * 30 * 1 * 2750
        # 30(ch) * 30 * 1 * 2750
        out = torch.permute(input, (0, 2, 1))
        # [batch, time', channel*n_filter]
        out = torch.matmul(out, self.A)
        return out

def cov_mat(data):
    batch_size, t, num_channels = data.shape
    data_centered = data - data.mean(dim=1, keepdim=True)
    covariance_matrices = torch.matmul(data_centered.transpose(1, 2), data_centered) / (t - 1)
    return covariance_matrices

def frobenius_distance(matrix_a, matrix_b):
    # 计算Frobenius距离
    if matrix_a.shape != matrix_b.shape:
        raise ValueError("Must have same shape")
    if type(matrix_a) == torch.Tensor:
        distance = torch.linalg.norm(matrix_a - matrix_b, 'fro')
    if type(matrix_a) == np.ndarray:
        distance = np.linalg.norm(matrix_a - matrix_b, 'fro')
    return distance

def top1_accuracy(dis_mat, labels):
    # Get the predicted classes (indices of the 到原型最小距离 in each row)
    predicted_classes = torch.argmax(dis_mat, dim=1)
    
    # Flatten labels to match predicted_classes shape
    labels = labels.view(-1)
    
    # Calculate the number of correct predictions
    correct_predictions = (predicted_classes == labels).sum().item()
    
    # Calculate accuracy
    accuracy = correct_predictions / labels.size(0)
    
    return accuracy

def init_proto(dim, n_class):
    return nn.Parameter(torch.randn((n_class, dim, dim)))

def loss_proto(cov, label, protos):
    # print(cov.shape)
    # print(label.shape)
    # print(protos.shape)
    batch_size = cov.shape[0]
    loss = 0
    dis_mat = torch.zeros((batch_size, protos.shape[0])).to('cuda')
    for i, cov_i in enumerate(cov):
        for j, pro_j in enumerate(protos):
            dis_mat[i][j] = -frobenius_distance(cov_i, pro_j)
    # print(dis_mat)
    acc = top1_accuracy(dis_mat, label)
    dis_mat = F.log_softmax(dis_mat, dim=-1)
    NLLLoss = nn.NLLLoss()
    # print(dis_mat)
    loss = NLLLoss(dis_mat, label)
    # print(loss)

    return loss, acc



class DualModel_PL(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.channelwiseEncoder = channel_MLLA(
            standard_channels=cfg.data_2.channels, 
            patch_size=cfg.channel_encoder.patch_size,
            hidden_dim=cfg.channel_encoder.hidden_dim,
            out_dim=cfg.channel_encoder.out_dim,
            patch_stride=cfg.channel_encoder.patch_stride)
        self.alignmentModule_1 = Channel_Alignment(cfg.data_1.n_channs*cfg.channel_encoder.out_dim, cfg.align.n_channel_uni)
        self.alignmentModule_2 = Channel_Alignment(cfg.data_2.n_channs*cfg.channel_encoder.out_dim, cfg.align.n_channel_uni)
        self.protos_1 = init_proto(dim=cfg.align.n_channel_uni, n_class=cfg.data_1.n_class)
        self.protos_2 = init_proto(dim=cfg.align.n_channel_uni, n_class=cfg.data_2.n_class)
        self.lr = cfg.train.lr
        self.wd = cfg.train.wd
        self.max_epochs = cfg.train.max_epochs
        self.restart_times = cfg.train.restart_times
        self.is_logger = cfg.log.is_logger
    def forward(self, batch, channel_names):
        feature = self.channelwiseEncoder(batch, channel_names)
        return feature 
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        # scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.max_epochs, gamma=0.8, last_epoch=-1, verbose=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.max_epochs // self.restart_times, eta_min=0,last_epoch=-1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    # remain to be implemented
    def training_step(self, batch, batch_idx):
        data_1, data_2 = batch
        x_1, y_1 = data_1
        x_2, y_2 = data_2
        fea_1 = self.forward(x_1, self.cfg.data_1.channels)
        fea_2 = self.forward(x_2, self.cfg.data_2.channels)
        fea_1 = self.alignmentModule_1(fea_1)
        fea_2 = self.alignmentModule_2(fea_2)
        cov_1 = cov_mat(fea_1)
        cov_2 = cov_mat(fea_2)
        # print(cov_1.shape, cov_2.shape)
        # print(y_1, y_2)
        loss_class_1, acc_1 = loss_proto(cov_1, y_1, self.protos_1)
        loss_class_2, acc_2 = loss_proto(cov_2, y_2, self.protos_2)
        # print(f'train/loss_class_1={loss_class_1},train/loss_class_2={loss_class_2},train/acc_1={acc_1},train/acc_2={acc_2}')
        self.log_dict({
                    'loss_class_1/train': loss_class_1, 
                    'loss_class_2/train': loss_class_2, 
                    'acc_1/train': acc_1, 
                    'acc_2/train': acc_2, 
                    'lr': self.optimizers().param_groups[-1]['lr']
                    },
                    logger=self.is_logger,
                    on_step=False, on_epoch=True, prog_bar=True)
        # fea.shape = [channel, n_filter, time']
        # self.criterion.to(data.device)   # put it in the loss function

        loss = loss_class_1 + loss_class_2
        return loss
    
    def validation_step(self, batch, batch_idx):
        data_1, data_2 = batch
        x_1, y_1 = data_1
        x_2, y_2 = data_2
        fea_1 = self.forward(x_1, self.cfg.data_1.channels)
        fea_2 = self.forward(x_2, self.cfg.data_2.channels)
        fea_1 = self.alignmentModule_1(fea_1)
        fea_2 = self.alignmentModule_2(fea_2)
        cov_1 = cov_mat(fea_1)
        cov_2 = cov_mat(fea_2)
        # print(cov_1.shape, cov_2.shape)
        # print(y_1, y_2)
        loss_class_1, acc_1 = loss_proto(cov_1, y_1, self.protos_1)
        loss_class_2, acc_2 = loss_proto(cov_2, y_2, self.protos_2)
        # print(f'test/loss_class_1={loss_class_1},test/loss_class_2={loss_class_2},test/acc_1={acc_1},test/acc_2={acc_2}')
        self.log_dict({
                    'loss_class_1/val': loss_class_1, 
                    'loss_class_2/val': loss_class_2, 
                    'acc_1/val': acc_1, 
                    'acc_2/val': acc_2, 
                    'lr/val': self.optimizers().param_groups[-1]['lr']
                    },
                    logger=self.is_logger,
                    on_step=False, on_epoch=True, prog_bar=True)
        # fea.shape = [channel, n_filter, time']
        # self.criterion.to(data.device)   # put it in the loss function

        loss = loss_class_1 + loss_class_2
        return loss
    
    def predict_step(self, batch, batch_idx):
        data_1, data_2 = batch
        x_1, y_1 = data_1
        x_2, y_2 = data_2
        fea_1 = self.forward(x_1, self.cfg.data_1.channels)
        fea_2 = self.forward(x_2, self.cfg.data_2.channels)
        return fea_1