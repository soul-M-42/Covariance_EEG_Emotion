import torch
import pytorch_lightning as pl
import torch.nn as nn
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
import random
from src.model.CNN_Attention import cnn_PatchTST, cnn_MLLA, Conv_att_simple_mlp
from src.model.Channel_MLP import Channel_mlp_CNN
from src.model.PatchTSTsingle import PatchTST_single_backbone
from src.model.MLLA_new import channel_MLLA
from src.loss.loss import SimCLRLoss
from src.loss.CDA_loss import CDALoss
from src.utils import report_vram

class MultiModel_PL(pl.LightningModule):
    def __init__(self, cfg=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_fea = False
        
        # load channel project matrix
        cha_weight_dir = '/mnt/dataset1/ws2319/workspace/ESI_pipelines/qingzhu/leadfield_fsaverage_standard_1020.npz'
        cha_weight_data = np.load(cha_weight_dir)
        leadfield =cha_weight_data['leadfield']
        self.uni_channel_names = cha_weight_data['channel_names']
        cha_weight_data.close()
        pinv_leadfield = np.linalg.pinv(leadfield)
        self.leadfield = torch.Tensor(leadfield)
        self.pinv_leadfield = torch.Tensor(pinv_leadfield)
        if(cfg.model.encoder == 'cnn'):
            self.cnn_encoder = Conv_att_simple_mlp(cfg.model.cnn.n_timeFilters,
                                               cfg.model.cnn.timeFilterLen,
                                               cfg.model.cnn.n_msFilters,
                                               cfg.model.cnn.msFilter_timeLen,
                                               cfg.model.cnn.n_channs,
                                               cfg.model.cnn.dilation_array,
                                               cfg.model.cnn.seg_att, 
                                               cfg.model.cnn.avgPoolLen,
                                               cfg.model.cnn.timeSmootherLen,
                                               cfg.model.cnn.multiFact,
                                               cfg.model.cnn.stratified, 
                                               cfg.model.cnn.activ,
                                               cfg.model.cnn.temp,
                                               cfg.model.cnn.saveFea,
                                               cfg.model.cnn.has_att,
                                               cfg.model.cnn.extract_mode,
                                               cfg.model.cnn.global_att,
                                               c_mlps = [Channel_mlp_CNN(cfg_i.n_channs, cfg.model.cnn.n_channs) for cfg_i in cfg.data_cfg_list])

        if(cfg.model.encoder == 'TST_single'):
            self.c_mlps = [Channel_mlp_CNN(cfg_i.n_channs, cfg.model.TST_single.cnn.n_channs) for cfg_i in cfg.data_cfg_list]
            self.patchTST = PatchTST_single_backbone(c_in=1,
                                              context_window=cfg.data_0.timeLen * cfg.data_0.fs,
                                              patch_len=cfg.model.TST_single.patch_len,
                                              stride=cfg.model.TST_single.patch_stride,
                                              d_model=cfg.model.TST_single.cnn.n_timeFilters,
                                              n_heads=cfg.model.TST_single.n_heads)
            self.cnn_encoder = cnn_PatchTST(cfg.model.TST_single.cnn.n_timeFilters,
                                               cfg.model.TST_single.cnn.timeFilterLen,
                                               cfg.model.TST_single.cnn.n_msFilters,
                                               cfg.model.TST_single.cnn.msFilter_timeLen,
                                               cfg.model.TST_single.cnn.n_channs,
                                               cfg.model.TST_single.cnn.dilation_array,
                                               cfg.model.TST_single.cnn.seg_att, 
                                               cfg.model.TST_single.cnn.avgPoolLen,
                                               cfg.model.TST_single.cnn.timeSmootherLen,
                                               cfg.model.TST_single.cnn.multiFact,
                                               cfg.model.TST_single.cnn.stratified, 
                                               cfg.model.TST_single.cnn.activ,
                                               cfg.model.TST_single.cnn.temp,
                                               cfg.model.TST_single.cnn.saveFea,
                                               cfg.model.TST_single.cnn.has_att,
                                               cfg.model.TST_single.cnn.extract_mode,
                                               cfg.model.TST_single.cnn.global_att)
        if(cfg.model.encoder == 'MLLA'):
            # self.c_mlps = nn.ModuleList([Channel_mlp_CNN(cfg_i.n_channs, cfg.model.MLLA.cnn.n_channs) for cfg_i in cfg.data_cfg_list])
            self.uni_mlp = Channel_mlp_CNN(len(self.uni_channel_names), cfg.model.MLLA.cnn.n_channs)
            self.MLLA = channel_MLLA(
                context_window=cfg.data_0.timeLen * cfg.data_0.fs,
                patch_size=cfg.model.MLLA.patch_size,
                hidden_dim=cfg.model.MLLA.hidden_dim,
                out_dim=cfg.model.MLLA.out_dim,
                depth=cfg.model.MLLA.depth,
                patch_stride=cfg.model.MLLA.patch_stride,
                n_heads=cfg.model.MLLA.n_heads)
            self.cnn_encoder = cnn_MLLA(cfg.model.MLLA.cnn.n_timeFilters,
                                               cfg.model.MLLA.cnn.timeFilterLen,
                                               cfg.model.MLLA.cnn.n_msFilters,
                                               cfg.model.MLLA.cnn.msFilter_timeLen,
                                               cfg.model.MLLA.cnn.n_channs,
                                               cfg.model.MLLA.cnn.dilation_array,
                                               cfg.model.MLLA.cnn.seg_att, 
                                               cfg.model.MLLA.cnn.avgPoolLen,
                                               cfg.model.MLLA.cnn.timeSmootherLen,
                                               cfg.model.MLLA.cnn.multiFact,
                                               cfg.model.MLLA.cnn.stratified, 
                                               cfg.model.MLLA.cnn.activ,
                                               cfg.model.MLLA.cnn.temp,
                                               cfg.model.MLLA.cnn.saveFea,
                                               cfg.model.MLLA.cnn.has_att,
                                               cfg.model.MLLA.cnn.extract_mode,
                                               cfg.model.MLLA.cnn.global_att)
        self.clisa_loss = SimCLRLoss(cfg.train.loss.temp)
        self.cda_loss = CDALoss(cfg)
        self.channel_projection_matrix = [[None] * len(self.cfg.data_cfg_list)][0]
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.train.lr, weight_decay=self.cfg.train.wd)
        return {'optimizer': optimizer}
    
    def forward(self, x, dataset=0):
        if(self.cfg.model.encoder == 'cnn'):
            if self.save_fea:
                self.cnn_encoder.saveFea = True
            x = self.cnn_encoder(x, dataset)
            return x
        if(self.cfg.model.encoder == 'TST_single'):
            x = x.squeeze(1)
            x = self.patchTST(x)
            x = torch.permute(x, (0, 2, 1, 3))
            x = self.c_mlps[dataset](x)
            if self.save_fea:
                self.cnn_encoder.saveFea = True
            x = self.cnn_encoder(x)
            return x
        if(self.cfg.model.encoder == 'MLLA'):
            x = self.MLLA(x)
            x = torch.permute(x, (0, 3, 1, 2))
            # x = self.c_mlps[dataset](x)
            x = self.uni_mlp(x)
            fea_cov = x
            if self.save_fea:
                self.cnn_encoder.saveFea = True
            x = self.cnn_encoder(x)
            return x, fea_cov
    
    def training_step(self, batch, batch_idx):
        loss = 0
        x_list, y_list = batch
        n_dataset = len(x_list)
        # x_list = [x_i[0] for x_i in x_list]  # 提取数据
        x_list = [self.channel_project(x_list[i][0], self.cfg.data_cfg_list[i].channels, dataset=i) for i in range(n_dataset-1)]  # 提取数据
        fea_clisa = []
        fea_cov = []

        for dataset in range(n_dataset-1):
            fea_clisa_i, fea_cov_i = self.forward(x_list[dataset], dataset)
            fea_clisa.append(fea_clisa_i)
            fea_cov.append(fea_cov_i)

        # 计算损失
        if self.cfg.train.loss.clisa_loss:
            for dataset, fea_clisa_i in enumerate(fea_clisa):
                clisa_loss_i, logits_i, labels_i, (acc_1, acc_5) = self.clisa_loss(fea_clisa_i)
                loss += clisa_loss_i

            # 记录日志
                self.log_dict({
                    f'loss_clisa_{self.cfg.data_cfg_list[dataset].dataset_name}/train': clisa_loss_i,
                    # f'acc1_{self.cfg.data_cfg_list[dataset].dataset_name}/train': acc_1,
                    # f'acc5_{self.cfg.data_cfg_list[dataset].dataset_name}/train': acc_5,
                }, on_step=False, on_epoch=True, prog_bar=True)
        if self.cfg.train.loss.CDA_loss:
            cda_loss = self.cda_loss(fea_cov) * self.cfg.train.loss.CDA_factor
            loss += cda_loss
            self.log_dict({
                f'loss_cda/train': cda_loss,
            }, on_step=False, on_epoch=True, prog_bar=True)


        # 显式释放显存（可选）
        del fea_clisa, fea_cov, clisa_loss_i, logits_i, labels_i, acc_1, acc_5
        torch.cuda.empty_cache()  # 清空缓存

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = 0
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        # 用来临时指定predict时用谁的mlp。-1即为未训练的随机mlp。（原本是作为微调基底）
        x = self.channel_project(x, self.cfg.data_val.channels, dataset=-1)
        fea_clisa_i, fea_cov_i = self.forward(x, 0)
        return fea_clisa_i
    
    def channel_project(self, data, cha_source, dataset=0):
        device = data.device
        
        # 统一转换为大写进行匹配（根据你的实际代码调整）
        standard_channels_upper = [name.upper() for name in self.uni_channel_names]
        cha_source_upper = [name.upper() for name in cha_source]
        
        # 筛选有效通道
        valid_indices = []
        valid_source_indices = []
        for idx, name in enumerate(cha_source_upper):
            if name in standard_channels_upper:
                valid_indices.append(standard_channels_upper.index(name))
                valid_source_indices.append(idx)
        if not valid_indices:
            raise ValueError("没有有效的通道可以映射")
        
        # 验证数据维度
        batch_size, _, n_channel_source, n_time = data.shape
        assert n_channel_source == len(cha_source), "输入通道数不匹配"
        
        if self.channel_projection_matrix[dataset] is None:
            # 将leadfield转换为张量并移动到对应设备
            leadfield = torch.as_tensor(self.leadfield, device=device, dtype=data.dtype)  # (94, 61452)
            pinv_leadfield = torch.as_tensor(self.pinv_leadfield, device=device, dtype=data.dtype)  # (94, 61452)
            
            projection_matrix = leadfield[valid_indices, :]  # [k, 61452]
            combined_projection = projection_matrix @ pinv_leadfield  # [k, 94]
            self.channel_projection_matrix[dataset] = combined_projection
        else:
            combined_projection = self.channel_projection_matrix[dataset]
        
        # 提取有效通道数据
        data_valid = data[:, :, valid_source_indices, :]  # [batch, 1, k, time]
        
        # --- 合并后的投影计算 ---
        # 重塑数据为 [batch*time, k]
        data_reshaped = data_valid.permute(0, 3, 1, 2).reshape(-1, len(valid_indices))  # [batch*time, k]

        
        # 单次矩阵乘法 [batch*time, k] @ [k, 94] → [batch*time, 94]
        back_projection = data_reshaped @ combined_projection
        
        # 恢复最终形状 [batch_size, 1, 94, n_time]
        result = back_projection.reshape(batch_size, n_time, 1, 94).permute(0, 2, 3, 1)
        
        return result