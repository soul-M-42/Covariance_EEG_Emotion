import torch
import pytorch_lightning as pl
import torch.nn as nn
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import matplotlib.pyplot as plt
import itertools
import time
import random
from src.model.CNN_Attention import Conv_att_simple_new, Conv_att_simple_mlp
from src.model.Channel_MLP import Channel_mlp_CNN
from src.model.PatchTST import PatchTST_backbone
from src.model.PatchTSTsingle import PatchTST_single_backbone
from src.loss.loss import SimCLRLoss

class MultiModel_PL(pl.LightningModule):
    def __init__(self, cfg=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_fea = False
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
            self.cnn_encoder = Conv_att_simple_new(cfg.model.TST_single.cnn.n_timeFilters,
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
        self.clisa_loss = SimCLRLoss(cfg.train.loss.temp)
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
    
    def training_step(self, batch, batch_idx):
        loss = 0
        x_list, y_list = batch
        x_list = [x_i[0] for x_i in x_list]
        # random_set = random.sample(range(len(x_list)-1), 3)
        # x_list = [x_list[i] for i in random_set]
        # y_list = [y_list[i] for i in random_set]
        fea_clisa_list = []
        for i in range(len(x_list)-1):
            fea = self.forward(x_list[i], i)
            fea_clisa_list.append(fea)

        if self.cfg.train.loss.clisa_loss:
            loss_clisa = [self.clisa_loss(fea_clisa_i) for fea_clisa_i in fea_clisa_list]
            for i, [clisa_loss_i, logits_i, labels_i, [acc_1, acc_5]] in enumerate(loss_clisa):
                loss += clisa_loss_i
                self.log_dict({
                    f'loss_clisa_{self.cfg.data_cfg_list[i].dataset_name}/train': clisa_loss_i,
                    f'acc1_{self.cfg.data_cfg_list[i].dataset_name}/train': acc_1,
                    f'acc5_{self.cfg.data_cfg_list[i].dataset_name}/train': acc_5,
                }, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = 0
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        # 用来临时指定predict时用谁的mlp。-1即为未训练的随机mlp。（原本是作为微调基底）
        fea = self.forward(x, 0)
        return fea